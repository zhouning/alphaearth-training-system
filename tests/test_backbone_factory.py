from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_build_backbone_preserves_legacy_prithvi_defaults():
    from geoadapter.models.backbone_factory import build_backbone
    from geoadapter.models.prithvi import PrithviBackbone

    spec = build_backbone(
        {
            "prithvi": {
                "pretrained": False,
                "checkpoint": None,
            }
        }
    )

    assert spec.name == "prithvi"
    assert isinstance(spec.model, PrithviBackbone)
    assert spec.feature_dim == 768
    assert spec.input_channels == 6
    assert spec.blocks is spec.model.blocks
    assert len(spec.blocks) == 12


def test_build_backbone_constructs_satmae_without_weights_for_unit_tests():
    from geoadapter.models.backbone_factory import build_backbone
    from geoadapter.models.satmae import SatMAEBackbone

    spec = build_backbone(
        {
            "backbone": {
                "name": "satmae_vit_base",
                "family": "satmae",
                "pretrained": False,
                "checkpoint": None,
                "input_channels": 10,
                "embed_dim": 128,
                "depth": 2,
                "num_heads": 4,
                "patch_size": 16,
            }
        }
    )

    assert spec.name == "satmae_vit_base"
    assert isinstance(spec.model, SatMAEBackbone)
    assert spec.feature_dim == 128
    assert spec.input_channels == 10
    assert len(spec.blocks) == 2

    x = torch.randn(2, 10, 64, 64)
    features = spec.model(x)
    assert features.shape == (2, 128)
    assert all(not param.requires_grad for param in spec.model.parameters())


def test_satmae_pretrained_requires_existing_checkpoint(tmp_path: Path):
    from geoadapter.models.backbone_factory import build_backbone

    missing = tmp_path / "missing_satmae.pth"
    with pytest.raises(FileNotFoundError, match="SatMAE checkpoint not found"):
        build_backbone(
            {
                "backbone": {
                    "name": "satmae_vit_base",
                    "family": "satmae",
                    "pretrained": True,
                    "checkpoint": str(missing),
                    "input_channels": 10,
                    "embed_dim": 128,
                    "depth": 2,
                    "num_heads": 4,
                    "patch_size": 16,
                }
            }
        )


def test_unknown_backbone_family_fails_clearly():
    from geoadapter.models.backbone_factory import build_backbone

    with pytest.raises(ValueError, match="unsupported backbone family"):
        build_backbone(
            {
                "backbone": {
                    "name": "x",
                    "family": "not_a_backbone",
                    "pretrained": False,
                }
            }
        )
