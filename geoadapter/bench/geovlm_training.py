from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

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
    probe_indices_by_class: dict[str, tuple[int, ...]]
    probe_sha256: str


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
