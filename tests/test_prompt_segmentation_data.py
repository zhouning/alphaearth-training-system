from pathlib import Path

import pytest
import torch

from geoadapter.data.prompt_segmentation import (
    LANDCOVERAI_CLASSES,
    PROMPT_TARGET_CLASS_IDS,
    load_prompt_config,
    multiclass_to_binary,
    normalize_landcoverai_image,
    prompt_batch_from_class_names,
    sample_prompt_batch,
)


CONFIG = Path("geoadapter/bench/configs/geovlm_prompts.yaml")


def test_landcoverai_taxonomy_matches_torchgeo():
    assert LANDCOVERAI_CLASSES == (
        "background",
        "building",
        "woodland",
        "water",
        "road",
    )
    assert PROMPT_TARGET_CLASS_IDS == {"building": 1, "water": 3, "road": 4}


def test_multiclass_to_binary_uses_requested_official_index():
    mask = torch.tensor([[0, 1, 3], [4, 2, 1]])
    assert multiclass_to_binary(mask, 1).tolist() == [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert multiclass_to_binary(mask, 3).sum().item() == 1
    assert multiclass_to_binary(mask, 4).sum().item() == 1


def test_landcoverai_image_normalization_maps_bytes_to_unit_interval():
    image = torch.tensor([0.0, 127.5, 255.0]).reshape(3, 1, 1)
    normalized = normalize_landcoverai_image(image)
    assert torch.allclose(
        normalized,
        torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1),
    )


def test_landcoverai_image_normalization_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="0..255"):
        normalize_landcoverai_image(torch.full((3, 1, 1), 300.0))


def test_prompt_config_has_disjoint_seen_and_held_out_prompts():
    config = load_prompt_config(CONFIG)
    assert set(config.classes) == {"building", "road", "water"}
    for prompt_class in config.classes.values():
        assert prompt_class.training
        assert prompt_class.held_out
        assert set(prompt_class.training).isdisjoint(prompt_class.held_out)


def test_prompt_config_rejects_overlap(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text(
        """schema: paper12.geovlm_prompts.v1
language: en
classes:
  building:
    class_id: 1
    training: [find buildings]
    held_out: [find buildings]
  water:
    class_id: 3
    training: [find water]
    held_out: [identify open water]
  road:
    class_id: 4
    training: [find roads]
    held_out: [identify paved routes]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_prompt_config(path)


def test_sample_prompt_batch_builds_binary_targets_and_prompts():
    masks = torch.stack(
        [
            torch.tensor([[1, 0], [0, 0]]),
            torch.tensor([[3, 3], [0, 0]]),
            torch.tensor([[4, 0], [4, 0]]),
        ]
    )
    config = load_prompt_config(CONFIG)
    batch = sample_prompt_batch(
        masks,
        config,
        generator=torch.Generator().manual_seed(7),
        empty_cap=0.25,
    )
    assert batch.class_ids.shape == (3,)
    assert batch.targets.shape == (3, 2, 2)
    assert len(batch.prompts) == 3
    assert batch.targets.dtype == torch.float32
    assert batch.voluntary_empty_count <= int(len(masks) * 0.25)


def test_sample_prompt_batch_rejects_invalid_empty_cap():
    config = load_prompt_config(CONFIG)
    with pytest.raises(ValueError, match="empty_cap"):
        sample_prompt_batch(torch.zeros(1, 2, 2), config, empty_cap=1.5)


def test_prompt_batch_from_class_names_uses_explicit_schedule():
    masks = torch.stack(
        [
            torch.tensor([[1, 0], [0, 0]]),
            torch.tensor([[3, 3], [0, 0]]),
            torch.tensor([[4, 0], [4, 0]]),
        ]
    )
    config = load_prompt_config(CONFIG)

    batch = prompt_batch_from_class_names(
        masks,
        ("water", "water", "road"),
        config,
        generator=torch.Generator().manual_seed(7),
    )

    assert batch.class_names == ("water", "water", "road")
    assert batch.class_ids.tolist() == [3, 3, 4]
    assert batch.empty_count == 1
    assert batch.voluntary_empty_count == 1
    assert batch.targets.flatten(1).sum(dim=1).tolist() == [0.0, 2.0, 2.0]
    assert all(
        prompt in config.classes[class_name].training
        for class_name, prompt in zip(batch.class_names, batch.prompts)
    )


def test_prompt_batch_from_class_names_rejects_wrong_schedule_length():
    config = load_prompt_config(CONFIG)

    with pytest.raises(ValueError, match="class_names"):
        prompt_batch_from_class_names(
            torch.zeros(2, 2, 2),
            ("building",),
            config,
        )


def test_prompt_batch_from_class_names_rejects_unsupported_class():
    config = load_prompt_config(CONFIG)

    with pytest.raises(ValueError, match="unsupported"):
        prompt_batch_from_class_names(
            torch.zeros(1, 2, 2),
            ("woodland",),
            config,
        )
