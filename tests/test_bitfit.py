import pytest
import torch
import torch.nn as nn

from geoadapter.adapters.bitfit import configure_bitfit


class TestBitFit:
    def test_only_biases_and_layernorm_weights_trainable(self):
        m = nn.Sequential(
            nn.Linear(8, 16, bias=True),
            nn.LayerNorm(16),
            nn.Linear(16, 4, bias=True),
        )
        configure_bitfit(m)
        for name, p in m.named_parameters():
            if "bias" in name:
                assert p.requires_grad, f"{name} should be trainable"
            elif "weight" in name and "LayerNorm" in str(type(m._modules[name.split(".")[0]])):
                # LN weight unfrozen for autograd-shape correctness
                assert p.requires_grad, f"LayerNorm weight {name} should be trainable"
            elif name.endswith(".weight") and isinstance(
                dict(m.named_modules())[name.rsplit(".", 1)[0]], nn.LayerNorm
            ):
                assert p.requires_grad
            else:
                # Linear weights frozen
                if "bias" not in name:
                    pass  # we check the freeze case below explicitly

        # Explicit freeze checks
        assert not m[0].weight.requires_grad
        assert not m[2].weight.requires_grad
        # Bias trainable
        assert m[0].bias.requires_grad
        assert m[2].bias.requires_grad
        # LN weight + bias trainable
        assert m[1].weight.requires_grad
        assert m[1].bias.requires_grad

    def test_backward_works_with_frozen_weights(self):
        """The autograd shape-mismatch bug from LoveDA Phase 2 must not return."""
        m = nn.Sequential(
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.Linear(16, 4),
        )
        configure_bitfit(m)
        x = torch.randn(2, 8)
        out = m(x).sum()
        out.backward()  # Must not raise NativeLayerNormBackward shape mismatch.
        # Frozen weights should have None grad; trainable params should have a grad.
        assert m[0].weight.grad is None
        assert m[0].bias.grad is not None
        assert m[1].weight.grad is not None
        assert m[1].bias.grad is not None

    def test_handles_layernorm_without_affine(self):
        m = nn.Sequential(
            nn.Linear(8, 16),
            nn.LayerNorm(16, elementwise_affine=False),
            nn.Linear(16, 4),
        )
        configure_bitfit(m)  # Must not crash on the no-affine LN.
        # The trainable set is just the two Linear biases.
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        assert "0.bias" in trainable
        assert "2.bias" in trainable
