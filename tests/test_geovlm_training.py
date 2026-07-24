import hashlib
import json

import pytest
import torch

from geoadapter.bench.geovlm_training import (
    CLASS_NAMES,
    reserve_training_probe,
    scan_target_present_pool,
)


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
