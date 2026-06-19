from __future__ import annotations

import pytest
import torch

from geoadapter.adapters.learned_channel_bridge import LearnedChannelBridgeAdapter
from geoadapter.adapters.zero_pad import ZeroPadAdapter


@pytest.mark.parametrize("c_in", [3, 6, 10])
def test_learned_channel_bridge_output_shape(c_in: int):
    adapter = LearnedChannelBridgeAdapter(in_channels=c_in, out_channels=6)
    x = torch.randn(2, c_in, 64, 64)
    out = adapter(x)
    assert out.shape == (2, 6, 64, 64)


@pytest.mark.parametrize("c_in", [3, 6, 10])
def test_learned_channel_bridge_matches_zero_pad_at_init(c_in: int):
    adapter = LearnedChannelBridgeAdapter(in_channels=c_in, out_channels=6)
    baseline = ZeroPadAdapter(in_channels=c_in, out_channels=6)
    x = torch.randn(2, c_in, 32, 32)
    with torch.no_grad():
        assert torch.allclose(adapter(x), baseline(x), atol=1e-6)


def test_learned_channel_bridge_has_trainable_projection():
    adapter = LearnedChannelBridgeAdapter(in_channels=10, out_channels=6)
    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    assert n_trainable == 60


def test_learned_channel_bridge_can_use_non_prithvi_channel_signal():
    adapter = LearnedChannelBridgeAdapter(in_channels=10, out_channels=6)
    x = torch.zeros(1, 10, 4, 4)
    x[:, 9] = 2.0
    with torch.no_grad():
        adapter.projection.weight.zero_()
        adapter.projection.weight[0, 9, 0, 0] = 0.5
    out = adapter(x)
    assert torch.allclose(out[:, 0], torch.ones(1, 4, 4))
