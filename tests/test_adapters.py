import pytest
import torch
from geoadapter.adapters.base import ModalityAdapter
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.adapters.zero_pad import ZeroPadAdapter


class TestModalityAdapterInterface:
    def test_geo_adapter_is_modality_adapter(self):
        adapter = GeoAdapter(in_channels=4, out_channels=6)
        assert isinstance(adapter, ModalityAdapter)

    def test_zero_pad_is_modality_adapter(self):
        adapter = ZeroPadAdapter(in_channels=4, out_channels=6)
        assert isinstance(adapter, ModalityAdapter)


class TestGeoAdapter:
    @pytest.mark.parametrize("c_in", [2, 3, 4, 5, 10])
    def test_output_shape(self, c_in):
        adapter = GeoAdapter(in_channels=c_in, out_channels=6)
        x = torch.randn(2, c_in, 128, 128)
        out = adapter(x)
        assert out.shape == (2, 6, 128, 128)

    def test_trainable_param_count(self):
        adapter = GeoAdapter(in_channels=4, out_channels=6)
        n = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        assert n < 1000, f"GeoAdapter should be <1000 params, got {n}"

    def test_three_layers_present(self):
        adapter = GeoAdapter(in_channels=5, out_channels=6)
        assert hasattr(adapter, "channel_proj")
        assert hasattr(adapter, "channel_attn")
        assert hasattr(adapter, "spatial_refine")


class TestZeroPadAdapter:
    @pytest.mark.parametrize("c_in", [2, 3, 4, 5, 10])
    def test_output_shape(self, c_in):
        adapter = ZeroPadAdapter(in_channels=c_in, out_channels=6)
        x = torch.randn(2, c_in, 64, 64)
        out = adapter(x)
        assert out.shape == (2, 6, 64, 64)

    def test_no_trainable_params(self):
        adapter = ZeroPadAdapter(in_channels=4, out_channels=6)
        n = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        assert n == 0

    def test_preserves_existing_channels(self):
        adapter = ZeroPadAdapter(in_channels=3, out_channels=6)
        x = torch.ones(1, 3, 4, 4)
        out = adapter(x)
        assert torch.allclose(out[0, :3], x[0])
        assert torch.allclose(out[0, 3:], torch.zeros(3, 4, 4))
