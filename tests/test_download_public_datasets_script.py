from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "scripts"))


class TinyClassificationDataset:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        return torch.zeros(10, 64, 64), torch.tensor(idx % 10)


class TinySegmentationDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        return torch.zeros(3, 1024, 1024), torch.zeros(1024, 1024, dtype=torch.long)


def test_download_script_downloads_eurosat_train_and_test(monkeypatch, tmp_path, capsys):
    import download_public_datasets

    calls = []

    def fake_load_eurosat(*, root, modality, split):
        calls.append((Path(root), modality, split))
        return TinyClassificationDataset()

    monkeypatch.setattr(download_public_datasets, "load_eurosat", fake_load_eurosat)

    code = download_public_datasets.main([
        "--dataset", "eurosat",
        "--eurosat-root", str(tmp_path / "eurosat"),
        "--max-samples", "1",
    ])

    assert code == 0
    assert calls == [
        (tmp_path / "eurosat", "s2_full", "train"),
        (tmp_path / "eurosat", "s2_full", "test"),
    ]
    assert "EuroSAT train" in capsys.readouterr().out


def test_download_script_downloads_all_loveda_crossdomain_splits(monkeypatch, tmp_path):
    import download_public_datasets

    calls = []

    def fake_load_loveda(*, root, domain, split, max_samples):
        calls.append((Path(root), domain, split, max_samples))
        return TinySegmentationDataset()

    monkeypatch.setattr(download_public_datasets, "load_loveda", fake_load_loveda)

    code = download_public_datasets.main([
        "--dataset", "loveda",
        "--loveda-root", str(tmp_path / "loveda"),
        "--max-samples", "2",
    ])

    assert code == 0
    assert calls == [
        (tmp_path / "loveda", "urban", "train", 2),
        (tmp_path / "loveda", "rural", "train", 2),
        (tmp_path / "loveda", "urban", "val", 2),
        (tmp_path / "loveda", "rural", "val", 2),
    ]


def test_download_script_force_removes_existing_dataset_root(monkeypatch, tmp_path):
    import download_public_datasets

    root = tmp_path / "eurosat"
    root.mkdir()
    partial = root / "EuroSATallBands.zip"
    partial.write_text("not a zip")

    observed_roots = []

    def fake_download_eurosat(root_arg, max_samples):
        observed_roots.append(root_arg)
        assert root.exists()
        assert not partial.exists()

    monkeypatch.setattr(download_public_datasets, "download_eurosat", fake_download_eurosat)

    code = download_public_datasets.main([
        "--dataset", "eurosat",
        "--eurosat-root", str(root),
        "--force",
    ])

    assert code == 0
    assert observed_roots == [root]


def test_download_script_force_refuses_to_delete_repo_root():
    import download_public_datasets

    with pytest.raises(ValueError, match="Refusing"):
        download_public_datasets.main([
            "--dataset", "eurosat",
            "--eurosat-root", str(repo_root),
            "--force",
        ])
