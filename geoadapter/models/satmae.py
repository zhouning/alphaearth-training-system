from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class SatMAEBackbone(nn.Module):
    """SatMAE-compatible frozen ViT backbone for Paper12 second-backbone checks."""

    _KEY_MAP = {
        "attn.qkv.weight": "self_attn.in_proj_weight",
        "attn.qkv.bias": "self_attn.in_proj_bias",
        "attn.proj.weight": "self_attn.out_proj.weight",
        "attn.proj.bias": "self_attn.out_proj.bias",
        "mlp.fc1.weight": "linear1.weight",
        "mlp.fc1.bias": "linear1.bias",
        "mlp.fc2.weight": "linear2.weight",
        "mlp.fc2.bias": "linear2.bias",
        "patch_embed.proj.weight": "patch_embed.weight",
        "patch_embed.proj.bias": "patch_embed.bias",
    }

    def __init__(
        self,
        *,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        in_chans: int = 10,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if pretrained:
            if not checkpoint_path:
                raise FileNotFoundError(
                    "SatMAE checkpoint not found: no checkpoint path configured"
                )
            self._load_checkpoint(checkpoint_path)

        self._freeze_all()

    def _freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def _checkpoint_state(self, payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in ("model", "state_dict", "checkpoint"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            if all(torch.is_tensor(value) for value in payload.values()):
                return payload
        raise ValueError("SatMAE checkpoint does not contain a tensor state dict")

    def _normalize_key(self, key: str) -> str:
        for prefix in ("module.", "encoder.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]

        for satmae_suffix, torch_suffix in self._KEY_MAP.items():
            if key.endswith(satmae_suffix):
                key = key[: -len(satmae_suffix)] + torch_suffix
                break
        return key

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"SatMAE checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = self._checkpoint_state(payload)
        own_state = self.state_dict()
        loaded = 0

        for raw_key, tensor in state.items():
            key = self._normalize_key(raw_key)
            if key in own_state and own_state[key].shape == tensor.shape:
                own_state[key] = tensor
                loaded += 1

        if loaded == 0:
            raise ValueError(f"No compatible SatMAE tensors loaded from {path}")
        self.load_state_dict(own_state, strict=False)

    def forward(self, x: torch.Tensor, return_spatial: bool = False):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if return_spatial:
            return x[:, 1:], (h, w)
        return x[:, 0]
