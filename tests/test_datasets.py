import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from geoadapter.data.transforms import BandSelector, Normalize
from geoadapter.data.datasets import ModalityConfig


def install_fake_loveda(monkeypatch, dataset):
    ctor = MagicMock(return_value=dataset)
    torchgeo = types.ModuleType("torchgeo")
    torchgeo_datasets = types.ModuleType("torchgeo.datasets")
    torchgeo_datasets.LoveDA = ctor
    torchgeo.datasets = torchgeo_datasets
    monkeypatch.setitem(sys.modules, "torchgeo", torchgeo)
    monkeypatch.setitem(sys.modules, "torchgeo.datasets", torchgeo_datasets)
    return ctor


class TestBandSelector:
    def test_select_rgb(self):
        sel = BandSelector(indices=[3, 2, 1])
        x = torch.randn(13, 64, 64)
        out = sel(x)
        assert out.shape == (3, 64, 64)
        assert torch.allclose(out[0], x[3])

    def test_identity(self):
        sel = BandSelector(indices=None)
        x = torch.randn(6, 64, 64)
        out = sel(x)
        assert out.shape == (6, 64, 64)


class TestNormalize:
    def test_output_range(self):
        norm = Normalize(method="log1p")
        x = torch.randint(0, 10000, (5, 64, 64)).float()
        out = norm(x)
        assert out.min() >= -10  # z-scored, so can be negative
        assert not torch.isnan(out).any()


class TestModalityConfig:
    def test_presets_exist(self):
        for preset in ["s2_full", "rgb", "rgb_sar", "gf2", "sar_only"]:
            cfg = ModalityConfig(preset)
            assert cfg.c_in > 0
            assert cfg.name


class TestLoveDA:
    def test_load_loveda_returns_segmentation_dataset(self, monkeypatch):
        from geoadapter.data.datasets import load_loveda, _SegmentationDataset

        mock_lo = MagicMock()
        mock_lo.__len__.return_value = 100
        mock_lo.__getitem__.return_value = {
            "image": torch.randn(3, 1024, 1024),
            "mask": torch.zeros(1024, 1024, dtype=torch.long),
        }
        ctor = install_fake_loveda(monkeypatch, mock_lo)

        ds = load_loveda(root="/tmp/loveda", domain="urban", split="train")

        assert isinstance(ds, _SegmentationDataset)
        call_kwargs = ctor.call_args.kwargs
        assert call_kwargs["split"] == "train"
        assert call_kwargs["scene"] == ["urban"]
        assert call_kwargs["download"] is True

    def test_load_loveda_rural_uses_rural_scene(self, monkeypatch):
        from geoadapter.data.datasets import load_loveda

        mock_lo = MagicMock()
        mock_lo.__len__.return_value = 50
        mock_lo.__getitem__.return_value = {
            "image": torch.randn(3, 1024, 1024),
            "mask": torch.zeros(1024, 1024, dtype=torch.long),
        }
        ctor = install_fake_loveda(monkeypatch, mock_lo)

        load_loveda(root="/tmp/loveda", domain="rural", split="val")

        call_kwargs = ctor.call_args.kwargs
        assert call_kwargs["split"] == "val"
        assert call_kwargs["scene"] == ["rural"]

    def test_load_loveda_rejects_bad_domain(self):
        from geoadapter.data.datasets import load_loveda
        with pytest.raises(ValueError, match="domain"):
            load_loveda(root="/tmp/loveda", domain="suburban", split="train")

    def test_load_loveda_max_samples_subsamples(self, monkeypatch):
        from geoadapter.data.datasets import load_loveda

        mock_lo = MagicMock()
        mock_lo.__len__.return_value = 1000
        mock_lo.__getitem__.return_value = {
            "image": torch.randn(3, 1024, 1024),
            "mask": torch.zeros(1024, 1024, dtype=torch.long),
        }
        install_fake_loveda(monkeypatch, mock_lo)

        ds = load_loveda(root="/tmp/loveda", domain="urban", split="train", max_samples=200)

        assert len(ds) == 200

    def test_load_loveda_remaps_mask_values(self, monkeypatch):
        """LoveDA masks are {0=ignore, 1..7=classes}; trainer expects 255 and [0..6]."""
        from geoadapter.data.datasets import load_loveda

        mask_raw = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ], dtype=torch.long)
        mock_lo = MagicMock()
        mock_lo.__len__.return_value = 1
        mock_lo.__getitem__.return_value = {
            "image": torch.randn(3, 2, 4),
            "mask": mask_raw.clone(),
        }
        install_fake_loveda(monkeypatch, mock_lo)

        ds = load_loveda(root="/tmp/loveda", domain="urban", split="train")
        _, mask_out = ds[0]

        expected = torch.tensor([
            [255, 0, 1, 2],
            [3,   4, 5, 6],
        ], dtype=torch.long)
        assert torch.equal(mask_out, expected), f"got {mask_out.tolist()}"
