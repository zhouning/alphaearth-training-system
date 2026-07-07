from __future__ import annotations

from pathlib import Path

import pytest
import torch


repo_root = Path(__file__).resolve().parents[1]
EUROSAT_ROOT = repo_root / "data" / "eurosat"
LOVEDA_ROOT = repo_root / "data" / "weights" / "raw_data" / "loveda"


def require_dataset_cache(root: Path, dataset_name: str) -> None:
    if not root.exists() or not any(root.iterdir()):
        pytest.skip(
            f"{dataset_name} cache is not available at {root}; "
            "run scripts/download_public_datasets.py first"
        )


def require_torchgeo() -> None:
    pytest.importorskip("torchgeo", reason="torchgeo is required for realdata smoke tests")


@pytest.mark.realdata
def test_eurosat_loader_reads_real_cached_train_and_test_splits():
    from geoadapter.data.datasets import load_eurosat

    require_dataset_cache(EUROSAT_ROOT, "EuroSAT")
    require_torchgeo()

    train_ds = load_eurosat(root=str(EUROSAT_ROOT), modality="s2_full", split="train")
    test_ds = load_eurosat(root=str(EUROSAT_ROOT), modality="s2_full", split="test")

    assert len(train_ds) > 0
    assert len(test_ds) > 0

    image, label = train_ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 10
    assert image.ndim == 3
    assert int(label) in range(10)


@pytest.mark.realdata
@pytest.mark.parametrize(
    ("domain", "split"),
    [
        ("urban", "train"),
        ("rural", "train"),
        ("urban", "val"),
        ("rural", "val"),
    ],
)
def test_loveda_loader_reads_real_cached_crossdomain_splits(domain: str, split: str):
    from geoadapter.data.datasets import load_loveda

    require_dataset_cache(LOVEDA_ROOT, "LoveDA")
    require_torchgeo()

    ds = load_loveda(root=str(LOVEDA_ROOT), domain=domain, split=split, max_samples=2)

    assert len(ds) > 0
    image, mask = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape[0] == 3
    assert image.ndim == 3
    assert mask.ndim == 2
    assert mask.shape == image.shape[1:]

    valid = mask[mask != 255]
    assert valid.numel() > 0
    assert int(valid.min()) >= 0
    assert int(valid.max()) <= 6
