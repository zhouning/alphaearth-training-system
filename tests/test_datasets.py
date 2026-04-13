import torch
from geoadapter.data.transforms import BandSelector, Normalize
from geoadapter.data.datasets import ModalityConfig


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
