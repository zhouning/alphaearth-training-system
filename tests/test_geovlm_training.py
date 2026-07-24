import hashlib
import json

import pytest
import torch

from geoadapter.bench.geovlm_training import (
    AssignedPromptDataset,
    CLASS_NAMES,
    build_epoch_assignments,
    reserve_training_probe,
    scan_target_present_pool,
)
from geoadapter.data.prompt_segmentation import PROMPT_TARGET_CLASS_IDS


PROMPT_CLASS_IDS = tuple(PROMPT_TARGET_CLASS_IDS.values())


def _sample(*class_ids):
    image = torch.zeros(3, 2, 2)
    values = tuple(class_ids) or (0,)
    mask = torch.tensor((values * 4)[:4]).reshape(2, 2)
    return image, mask


def test_scan_target_present_pool_excludes_no_target_samples():
    dataset = [
        _sample(0, 2),
        _sample(1),
        _sample(3, 4),
        _sample(4),
    ]

    pool = scan_target_present_pool(dataset)

    assert CLASS_NAMES == ("building", "water", "road")
    assert pool.source_size == 4
    assert pool.excluded_no_target_count == 1
    assert pool.excluded_no_target_share == pytest.approx(0.25)
    assert [sample.source_index for sample in pool.samples] == [1, 2, 3]
    assert pool.samples[1].present_classes == ("water", "road")


def test_scan_target_present_pool_rejects_empty_pool():
    with pytest.raises(ValueError, match="target-present training pool is empty"):
        scan_target_present_pool([_sample(0), _sample(2)])


def test_scan_target_present_pool_rejects_missing_supported_class():
    with pytest.raises(
        ValueError,
        match="target-present training pool has no samples for: water",
    ):
        scan_target_present_pool([_sample(1), _sample(4)])


def test_reserve_training_probe_is_seed_deterministic_and_deduplicated():
    dataset = [
        _sample(1, 3, 4),
        _sample(1),
        _sample(3),
        _sample(4),
        _sample(1, 3, 4),
        _sample(1, 3),
        _sample(3, 4),
        _sample(1, 4),
        _sample(1, 3, 4),
    ]
    pool = scan_target_present_pool(dataset)

    first = reserve_training_probe(pool, seed=42)
    second = reserve_training_probe(pool, seed=42)

    assert first == second
    assert tuple(first.probe_indices_by_class) == CLASS_NAMES
    assert all(
        len(first.probe_indices_by_class[class_name]) == 2
        for class_name in CLASS_NAMES
    )
    assert len(first.probe_indices) == len(set(first.probe_indices)) <= 6
    assert len(first.probe_sha256) == 64
    assert set(first.probe_indices).isdisjoint(
        sample.source_index for sample in first.training_samples
    )
    assert len(first.training_samples) + len(first.probe_indices) == len(pool.samples)

    hash_payload = {
        class_name: list(first.probe_indices_by_class[class_name])
        for class_name in CLASS_NAMES
    }
    expected_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first.probe_sha256 == expected_hash


def test_reserve_training_probe_class_indices_are_immutable():
    pool = scan_target_present_pool(
        [
            _sample(1, 3, 4),
            _sample(1, 3, 4),
            _sample(1, 3, 4),
        ]
    )

    split = reserve_training_probe(pool, seed=42)

    with pytest.raises(TypeError):
        split.probe_indices_by_class["building"] = ()


def test_reserve_training_probe_uses_seed():
    pool = scan_target_present_pool(
        [
            _sample(1, 3, 4),
            _sample(1),
            _sample(3),
            _sample(4),
            _sample(1, 3, 4),
            _sample(1, 3),
            _sample(3, 4),
            _sample(1, 4),
            _sample(1, 3, 4),
        ]
    )

    seed_42 = reserve_training_probe(pool, seed=42)
    seed_43 = reserve_training_probe(pool, seed=43)

    assert (
        seed_42.probe_indices_by_class != seed_43.probe_indices_by_class
        or seed_42.probe_indices != seed_43.probe_indices
    )


def test_reserve_training_probe_rejects_nonpositive_count():
    pool = scan_target_present_pool(
        [_sample(1, 3, 4), _sample(1, 3, 4), _sample(1, 3, 4)]
    )

    with pytest.raises(ValueError, match="positives_per_class must be positive"):
        reserve_training_probe(pool, seed=42, positives_per_class=0)


def test_reserve_training_probe_rejects_insufficient_positives():
    pool = scan_target_present_pool([_sample(1), _sample(3), _sample(4)])

    with pytest.raises(
        ValueError,
        match="insufficient probe positives for: building, water, road",
    ):
        reserve_training_probe(pool, seed=42)


def test_reserve_training_probe_rejects_empty_remaining_training_pool():
    pool = scan_target_present_pool(
        [_sample(1, 3, 4), _sample(1, 3, 4)]
    )

    with pytest.raises(ValueError, match="target-present training pool is empty after"):
        reserve_training_probe(pool, seed=42)


def test_reserve_training_probe_rejects_missing_class_after_removal():
    pool = scan_target_present_pool(
        [
            _sample(1),
            _sample(1),
            _sample(1),
            _sample(3, 4),
            _sample(3, 4),
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "target-present training pool has no samples after probe removal for: "
            "water, road"
        ),
    ):
        reserve_training_probe(pool, seed=42)


def test_build_epoch_assignments_is_deterministic_balanced_and_capped():
    dataset = [
        _sample(class_id)
        for class_id in PROMPT_CLASS_IDS
        for _ in range(8)
    ]
    split = reserve_training_probe(scan_target_present_pool(dataset), seed=42)

    first = build_epoch_assignments(
        split,
        batch_size=4,
        empty_target_cap=0.25,
        seed=42,
    )
    second = build_epoch_assignments(
        split,
        batch_size=4,
        empty_target_cap=0.25,
        seed=42,
    )

    assert first == second
    assert len(first) == len(split.training_samples)
    for offset in range(0, len(first), 4):
        batch = first[offset : offset + 4]
        assert sum(assignment.empty_target for assignment in batch) <= len(batch) // 4

    present_by_index = {
        sample.source_index: sample.present_classes
        for sample in split.training_samples
    }
    assert all(
        assignment.empty_target
        == (assignment.class_name not in present_by_index[assignment.source_index])
        for assignment in first
    )
    nonempty_counts = [
        sum(
            not assignment.empty_target and assignment.class_name == class_name
            for assignment in first
        )
        for class_name in CLASS_NAMES
    ]
    assert max(nonempty_counts) - min(nonempty_counts) <= 1

    assigned_dataset = AssignedPromptDataset(dataset, first)
    image, mask, class_name = assigned_dataset[0]
    expected_image, expected_mask = dataset[first[0].source_index]
    assert len(assigned_dataset) == len(split.training_samples)
    assert torch.equal(image, expected_image)
    assert torch.equal(mask, expected_mask)
    assert class_name == first[0].class_name


@pytest.mark.parametrize(
    ("batch_size", "empty_target_cap", "message"),
    [
        (0, 0.25, "batch_size"),
        (-1, 0.25, "batch_size"),
        (4, -0.01, "empty_target_cap"),
        (4, 1.01, "empty_target_cap"),
    ],
)
def test_build_epoch_assignments_rejects_invalid_limits(
    batch_size,
    empty_target_cap,
    message,
):
    dataset = [_sample(1, 3, 4) for _ in range(8)]
    split = reserve_training_probe(scan_target_present_pool(dataset), seed=42)

    with pytest.raises(ValueError, match=message):
        build_epoch_assignments(
            split,
            batch_size=batch_size,
            empty_target_cap=empty_target_cap,
            seed=42,
        )


def test_build_epoch_assignments_has_no_empty_targets_without_negatives():
    dataset = [_sample(1, 3, 4) for _ in range(8)]
    split = reserve_training_probe(scan_target_present_pool(dataset), seed=42)

    assignments = build_epoch_assignments(
        split,
        batch_size=4,
        empty_target_cap=1.0,
        seed=42,
    )

    assert len(assignments) == len(split.training_samples)
    assert not any(assignment.empty_target for assignment in assignments)
