from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import yaml


LANDCOVERAI_CLASSES = ("background", "building", "woodland", "water", "road")
PROMPT_TARGET_CLASS_IDS = {"building": 1, "water": 3, "road": 4}


@dataclass(frozen=True)
class PromptClass:
    class_id: int
    training: tuple[str, ...]
    held_out: tuple[str, ...]


@dataclass(frozen=True)
class PromptConfig:
    schema: str
    language: str
    classes: Mapping[str, PromptClass]


@dataclass(frozen=True)
class PromptBatch:
    class_ids: torch.Tensor
    class_names: tuple[str, ...]
    prompts: tuple[str, ...]
    targets: torch.Tensor
    empty_count: int
    voluntary_empty_count: int


def load_prompt_config(path: str | Path) -> PromptConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prompt config must be a mapping")
    if payload.get("schema") != "paper12.geovlm_prompts.v1":
        raise ValueError("unsupported prompt schema")
    if payload.get("language") != "en":
        raise ValueError("MVP prompt language must be en")

    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, dict):
        raise ValueError("prompt classes must be a mapping")
    classes: dict[str, PromptClass] = {}
    for name, expected_id in PROMPT_TARGET_CLASS_IDS.items():
        raw = raw_classes.get(name)
        if not isinstance(raw, dict) or int(raw.get("class_id", -1)) != expected_id:
            raise ValueError(f"invalid class mapping for {name}")
        training = tuple(str(value).strip() for value in raw.get("training", ()))
        held_out = tuple(str(value).strip() for value in raw.get("held_out", ()))
        if not training or not held_out or any(not value for value in training + held_out):
            raise ValueError(f"empty prompt set for {name}")
        if set(training) & set(held_out):
            raise ValueError(f"training/held-out prompt overlap for {name}")
        classes[name] = PromptClass(expected_id, training, held_out)
    return PromptConfig(payload["schema"], payload["language"], classes)


def sample_prompt_batch(
    masks: torch.Tensor,
    config: PromptConfig,
    *,
    generator: torch.Generator | None = None,
    empty_cap: float = 0.25,
) -> PromptBatch:
    if masks.ndim != 3:
        raise ValueError("masks must have shape [B,H,W]")
    if not 0.0 <= empty_cap <= 1.0:
        raise ValueError("empty_cap must be between 0 and 1")

    names = tuple(PROMPT_TARGET_CLASS_IDS)
    batch_size = int(masks.shape[0])
    repeated = (names * ((batch_size + len(names) - 1) // len(names)))[:batch_size]
    permutation = torch.randperm(batch_size, generator=generator).tolist()
    proposed = [repeated[index] for index in permutation]
    voluntary_empty_budget = int(batch_size * empty_cap)
    voluntary_empty_count = 0
    selected_names: list[str] = []
    prompts: list[str] = []
    targets: list[torch.Tensor] = []
    for mask, proposed_name in zip(masks, proposed):
        validate_landcoverai_mask(mask)
        present = [
            name
            for name in names
            if mask.eq(PROMPT_TARGET_CLASS_IDS[name]).any()
        ]
        forced_empty = not present
        proposed_is_empty = proposed_name not in present
        if (
            proposed_is_empty
            and not forced_empty
            and voluntary_empty_count >= voluntary_empty_budget
        ):
            index = int(torch.randint(len(present), (), generator=generator))
            name = present[index]
        else:
            name = proposed_name
            if proposed_is_empty and not forced_empty:
                voluntary_empty_count += 1
        prompt_values = config.classes[name].training
        prompt_index = int(torch.randint(len(prompt_values), (), generator=generator))
        selected_names.append(name)
        prompts.append(prompt_values[prompt_index])
        targets.append(multiclass_to_binary(mask, PROMPT_TARGET_CLASS_IDS[name]))

    target_tensor = torch.stack(targets)
    return PromptBatch(
        class_ids=torch.tensor(
            [PROMPT_TARGET_CLASS_IDS[name] for name in selected_names],
            dtype=torch.long,
        ),
        class_names=tuple(selected_names),
        prompts=tuple(prompts),
        targets=target_tensor,
        empty_count=int(target_tensor.flatten(1).sum(dim=1).eq(0).sum()),
        voluntary_empty_count=voluntary_empty_count,
    )


def validate_landcoverai_mask(mask: torch.Tensor) -> None:
    values = {int(value) for value in torch.unique(mask).tolist()}
    invalid = sorted(values - set(range(len(LANDCOVERAI_CLASSES))))
    if invalid:
        raise ValueError(f"LandCoverAI mask contains invalid class ids: {invalid}")


def multiclass_to_binary(mask: torch.Tensor, class_id: int) -> torch.Tensor:
    if class_id not in PROMPT_TARGET_CLASS_IDS.values():
        raise ValueError(f"unsupported prompt target class id: {class_id}")
    validate_landcoverai_mask(mask)
    return mask.eq(class_id).to(dtype=torch.float32)


def normalize_landcoverai_image(image: torch.Tensor) -> torch.Tensor:
    image = image.to(dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("LandCoverAI image must have shape [3,H,W]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("LandCoverAI image values must be finite and in 0..255")
    return image / 255.0
