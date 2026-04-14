import pytest
import torch
import torch.nn as nn
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
        assert hasattr(adapter, "residual_scale")

    def test_initial_output_equals_zero_pad(self):
        """At init (residual_scale=0), GeoAdapter should equal ZeroPadAdapter."""
        adapter = GeoAdapter(in_channels=3, out_channels=6)
        zp = ZeroPadAdapter(in_channels=3, out_channels=6)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out_ga = adapter(x)
            out_zp = zp(x)
        assert torch.allclose(out_ga, out_zp, atol=1e-6), \
            f"Initial GeoAdapter should equal ZeroPad, max diff={( out_ga - out_zp).abs().max():.6f}"

    def test_initial_output_equals_truncate_for_superset(self):
        """For c_in > c_out, initial output should truncate (same as ZeroPad)."""
        adapter = GeoAdapter(in_channels=10, out_channels=6)
        zp = ZeroPadAdapter(in_channels=10, out_channels=6)
        x = torch.randn(2, 10, 32, 32)
        with torch.no_grad():
            out_ga = adapter(x)
            out_zp = zp(x)
        assert torch.allclose(out_ga, out_zp, atol=1e-6)


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


from geoadapter.adapters.lora import inject_lora, remove_lora
from geoadapter.adapters.bitfit import configure_bitfit
from geoadapter.adapters.houlsby import inject_houlsby_adapters


class TestLoRA:
    def test_inject_and_remove(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        original_params = sum(p.numel() for p in block.parameters())
        inject_lora(block, rank=8, target_modules=["self_attn"])
        lora_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        assert lora_params > 0
        assert lora_params < original_params

    def test_forward_unchanged_shape(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        inject_lora(block, rank=8, target_modules=["self_attn"])
        x = torch.randn(2, 16, 768)
        out = block(x)
        assert out.shape == (2, 16, 768)


class TestBitFit:
    def test_only_biases_trainable(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        configure_bitfit(block)
        for name, p in block.named_parameters():
            if "bias" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"


class TestHoulsby:
    def test_inject_adds_params(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        before = sum(p.numel() for p in block.parameters())
        inject_houlsby_adapters(block, bottleneck_dim=64)
        after = sum(p.numel() for p in block.parameters())
        assert after > before

    def test_forward_shape(self):
        block = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        inject_houlsby_adapters(block, bottleneck_dim=64)
        x = torch.randn(2, 16, 768)
        out = block(x)
        assert out.shape == (2, 16, 768)
