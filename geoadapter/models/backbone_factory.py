from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.models.satmae import SatMAEBackbone


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    model: nn.Module
    feature_dim: int
    input_channels: int
    blocks: nn.ModuleList


def _legacy_prithvi_config(global_cfg: dict[str, Any]) -> dict[str, Any]:
    return global_cfg.get("prithvi") or {"pretrained": True, "checkpoint": None}


def _build_prithvi(global_cfg: dict[str, Any]) -> BackboneSpec:
    cfg = _legacy_prithvi_config(global_cfg)
    model = PrithviBackbone(
        pretrained=bool(cfg.get("pretrained", True)),
        checkpoint_path=cfg.get("checkpoint"),
        embed_dim=int(cfg.get("embed_dim", 768)),
        depth=int(cfg.get("depth", 12)),
        num_heads=int(cfg.get("num_heads", 12)),
        in_chans=int(cfg.get("input_channels", cfg.get("in_chans", 6))),
        patch_size=int(cfg.get("patch_size", 16)),
    )
    return BackboneSpec(
        name="prithvi",
        model=model,
        feature_dim=model.embed_dim,
        input_channels=int(cfg.get("input_channels", cfg.get("in_chans", 6))),
        blocks=model.blocks,
    )


def _build_satmae(backbone_cfg: dict[str, Any]) -> BackboneSpec:
    model = SatMAEBackbone(
        pretrained=bool(backbone_cfg.get("pretrained", True)),
        checkpoint_path=backbone_cfg.get("checkpoint"),
        in_chans=int(backbone_cfg.get("input_channels", 10)),
        embed_dim=int(backbone_cfg.get("embed_dim", 768)),
        depth=int(backbone_cfg.get("depth", 12)),
        num_heads=int(backbone_cfg.get("num_heads", 12)),
        patch_size=int(backbone_cfg.get("patch_size", 16)),
    )
    return BackboneSpec(
        name=str(backbone_cfg.get("name", "satmae_vit_base")),
        model=model,
        feature_dim=model.embed_dim,
        input_channels=model.in_chans,
        blocks=model.blocks,
    )


def build_backbone(global_cfg: dict[str, Any]) -> BackboneSpec:
    backbone_cfg = global_cfg.get("backbone")
    if not backbone_cfg:
        return _build_prithvi(global_cfg)

    family = str(
        backbone_cfg.get("family", backbone_cfg.get("name", "prithvi"))
    ).lower()
    if family == "prithvi":
        merged = dict(global_cfg)
        merged["prithvi"] = backbone_cfg
        return _build_prithvi(merged)
    if family == "satmae":
        return _build_satmae(backbone_cfg)
    raise ValueError(f"unsupported backbone family: {family}")
