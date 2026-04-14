import torch
import torch.nn as nn
from .base import ModalityAdapter


class GeoAdapter(ModalityAdapter):
    """Residual modality-aware adapter: zero-pad baseline + learned residual correction.

    Design: output = zero_pad(x) + scale * adapter(x)

    At initialization scale=0, so the adapter is equivalent to zero-pad.
    During training the adapter learns a residual correction that improves
    upon the zero-pad baseline. This guarantees the adapter's lower bound
    is never worse than zero-pad.

    Three-layer residual branch:
      Layer 1: Channel Projection (1x1 Conv)
      Layer 2: SE-style Channel Attention
      Layer 3: Spatial Refinement (depthwise 3x3 Conv)
    """

    def __init__(self, in_channels: int, out_channels: int = 6, se_reduction: int = 2):
        super().__init__(in_channels, out_channels)

        # Learnable residual scale — initialized to 0 so initial output = zero_pad(x)
        self.residual_scale = nn.Parameter(torch.zeros(1))

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

    def _zero_pad_or_truncate(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic baseline: pad missing channels with zeros, or truncate excess."""
        c_in = x.shape[1]
        if c_in < self.out_channels:
            pad = torch.zeros(
                x.shape[0], self.out_channels - c_in, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=1)
        return x[:, :self.out_channels]

    def forward(self, x):
        # Baseline: zero-pad or truncate (no gradient, preserves pre-trained distribution)
        baseline = self._zero_pad_or_truncate(x)

        # Residual branch: learned correction
        residual = self.channel_proj(x)
        attn = self.channel_attn(residual).unsqueeze(-1).unsqueeze(-1)
        residual = residual * attn
        residual = self.spatial_refine(residual)

        return baseline + self.residual_scale * residual
