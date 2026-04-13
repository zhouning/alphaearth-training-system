import torch
import torch.nn as nn
from .base import ModalityAdapter


class GeoAdapter(ModalityAdapter):
    """Three-layer modality-aware adapter: Projection + SE Attention + Spatial Refinement."""

    def __init__(self, in_channels: int, out_channels: int = 6, se_reduction: int = 2):
        super().__init__(in_channels, out_channels)

        # Layer 1: Channel Projection
        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Layer 2: SE-style Channel Attention
        mid = max(1, out_channels // se_reduction)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, out_channels),
            nn.Sigmoid(),
        )

        # Layer 3: Spatial Refinement (depthwise conv)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.channel_proj(x)
        attn = self.channel_attn(x).unsqueeze(-1).unsqueeze(-1)
        x = x * attn
        x = self.spatial_refine(x)
        return x
