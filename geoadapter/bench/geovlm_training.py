from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from geoadapter.data.prompt_segmentation import (
    PROMPT_TARGET_CLASS_IDS,
    validate_landcoverai_mask,
)


CLASS_NAMES = tuple(PROMPT_TARGET_CLASS_IDS)


@dataclass(frozen=True)
class TargetPresentSample:
    source_index: int
    present_classes: tuple[str, ...]


@dataclass(frozen=True)
class TargetPresentPool:
    source_size: int
    samples: tuple[TargetPresentSample, ...]
    excluded_no_target_count: int

    @property
    def excluded_no_target_share(self) -> float:
        if self.source_size == 0:
            return 0.0
        return self.excluded_no_target_count / self.source_size


@dataclass(frozen=True)
class TrainingProbeSplit:
    training_samples: tuple[TargetPresentSample, ...]
    probe_indices: tuple[int, ...]
    probe_indices_by_class: Mapping[str, tuple[int, ...]]
    probe_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probe_indices_by_class",
            MappingProxyType(dict(self.probe_indices_by_class)),
        )


@dataclass(frozen=True)
class PromptAssignment:
    source_index: int
    class_name: str
    empty_target: bool


class AssignedPromptDataset:
    def __init__(self, dataset, assignments: tuple[PromptAssignment, ...]) -> None:
        self.dataset = dataset
        self.assignments = assignments

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, index: int):
        assignment = self.assignments[index]
        image, mask = self.dataset[assignment.source_index]
        return image, mask, assignment.class_name


def scan_target_present_pool(dataset) -> TargetPresentPool:
    source_size = len(dataset)
    samples = []
    for source_index in range(source_size):
        _, mask = dataset[source_index]
        validate_landcoverai_mask(mask)
        present_classes = tuple(
            class_name
            for class_name, class_id in PROMPT_TARGET_CLASS_IDS.items()
            if mask.eq(class_id).any()
        )
        if present_classes:
            samples.append(TargetPresentSample(source_index, present_classes))

    if not samples:
        raise ValueError("target-present training pool is empty")

    missing_classes = [
        class_name
        for class_name in CLASS_NAMES
        if not any(class_name in sample.present_classes for sample in samples)
    ]
    if missing_classes:
        raise ValueError(
            "target-present training pool has no samples for: "
            + ", ".join(missing_classes)
        )

    return TargetPresentPool(
        source_size=source_size,
        samples=tuple(samples),
        excluded_no_target_count=source_size - len(samples),
    )


def reserve_training_probe(
    pool: TargetPresentPool,
    *,
    seed: int,
    positives_per_class: int = 2,
) -> TrainingProbeSplit:
    if positives_per_class <= 0:
        raise ValueError("positives_per_class must be positive")

    permutation = torch.randperm(
        len(pool.samples),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    shuffled_samples = tuple(pool.samples[index] for index in permutation)
    candidates_by_class = {
        class_name: tuple(
            sample.source_index
            for sample in shuffled_samples
            if class_name in sample.present_classes
        )
        for class_name in CLASS_NAMES
    }
    insufficient_classes = [
        class_name
        for class_name in CLASS_NAMES
        if len(candidates_by_class[class_name]) < positives_per_class
    ]
    if insufficient_classes:
        raise ValueError(
            "insufficient probe positives for: " + ", ".join(insufficient_classes)
        )

    probe_indices_by_class = {
        class_name: candidates_by_class[class_name][:positives_per_class]
        for class_name in CLASS_NAMES
    }
    selected_indices = {
        source_index
        for class_name in CLASS_NAMES
        for source_index in probe_indices_by_class[class_name]
    }
    probe_indices = tuple(
        sample.source_index
        for sample in shuffled_samples
        if sample.source_index in selected_indices
    )
    training_samples = tuple(
        sample
        for sample in pool.samples
        if sample.source_index not in selected_indices
    )

    if not training_samples:
        raise ValueError("target-present training pool is empty after probe removal")
    missing_training_classes = [
        class_name
        for class_name in CLASS_NAMES
        if not any(
            class_name in sample.present_classes for sample in training_samples
        )
    ]
    if missing_training_classes:
        raise ValueError(
            "target-present training pool has no samples after probe removal for: "
            + ", ".join(missing_training_classes)
        )

    hash_payload = {
        class_name: list(probe_indices_by_class[class_name])
        for class_name in CLASS_NAMES
    }
    serialized = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
    probe_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return TrainingProbeSplit(
        training_samples=training_samples,
        probe_indices=probe_indices,
        probe_indices_by_class=probe_indices_by_class,
        probe_sha256=probe_sha256,
    )


def _balanced_names(
    count: int,
    names: tuple[str, ...],
    generator: torch.Generator,
) -> list[str]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return []
    if not names:
        raise ValueError("names must not be empty when count is positive")

    permutation = torch.randperm(len(names), generator=generator).tolist()
    shuffled_names = [names[index] for index in permutation]
    return [shuffled_names[index % len(shuffled_names)] for index in range(count)]


def _draw_source(
    candidates: tuple[int, ...],
    generator: torch.Generator,
) -> int:
    if not candidates:
        raise ValueError("source candidates must not be empty")
    index = int(torch.randint(len(candidates), (), generator=generator))
    return candidates[index]


def _floor_binary_float_product(size: int, cap: float) -> int:
    numerator, denominator = float(cap).as_integer_ratio()
    return (size * numerator) // denominator


def build_epoch_assignments(
    split: TrainingProbeSplit,
    *,
    batch_size: int,
    empty_target_cap: float,
    seed: int,
) -> tuple[PromptAssignment, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= empty_target_cap <= 1.0:
        raise ValueError("empty_target_cap must be between 0 and 1")

    present_by_index = {
        sample.source_index: sample.present_classes
        for sample in split.training_samples
    }
    positive_candidates = {
        class_name: tuple(
            source_index
            for source_index, present_classes in present_by_index.items()
            if class_name in present_classes
        )
        for class_name in CLASS_NAMES
    }
    missing_classes = [
        class_name
        for class_name in CLASS_NAMES
        if not positive_candidates[class_name]
    ]
    if missing_classes:
        raise ValueError(
            "training split has no positive candidates for: "
            + ", ".join(missing_classes)
        )

    negative_candidates = {
        class_name: tuple(
            source_index
            for source_index, present_classes in present_by_index.items()
            if class_name not in present_classes
        )
        for class_name in CLASS_NAMES
    }
    negative_names = tuple(
        class_name
        for class_name in CLASS_NAMES
        if negative_candidates[class_name]
    )

    epoch_size = len(split.training_samples)
    batch_sizes = [
        min(batch_size, epoch_size - offset)
        for offset in range(0, epoch_size, batch_size)
    ]
    empty_counts = [
        _floor_binary_float_product(size, empty_target_cap)
        if negative_names
        else 0
        for size in batch_sizes
    ]
    total_empty_count = sum(empty_counts)
    generator = torch.Generator().manual_seed(seed)
    nonempty_names = _balanced_names(
        epoch_size - total_empty_count,
        CLASS_NAMES,
        generator,
    )
    empty_names = _balanced_names(
        total_empty_count,
        negative_names,
        generator,
    )

    assignments: list[PromptAssignment] = []
    nonempty_offset = 0
    empty_offset = 0
    for size, empty_count in zip(batch_sizes, empty_counts):
        batch: list[PromptAssignment] = []
        for name in nonempty_names[
            nonempty_offset : nonempty_offset + size - empty_count
        ]:
            batch.append(
                PromptAssignment(
                    source_index=_draw_source(positive_candidates[name], generator),
                    class_name=name,
                    empty_target=False,
                )
            )
        nonempty_offset += size - empty_count

        for name in empty_names[empty_offset : empty_offset + empty_count]:
            batch.append(
                PromptAssignment(
                    source_index=_draw_source(negative_candidates[name], generator),
                    class_name=name,
                    empty_target=True,
                )
            )
        empty_offset += empty_count

        permutation = torch.randperm(size, generator=generator).tolist()
        assignments.extend(batch[index] for index in permutation)

    return tuple(assignments)
