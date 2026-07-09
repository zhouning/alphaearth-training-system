import math

import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MultiLabelHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 19):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class SegmentationHead(nn.Module):
    """Patch-token segmentation decoder with a linear default and optional conv path."""

    def __init__(
        self,
        in_dim: int = 768,
        num_classes: int = 2,
        patch_size: int = 16,
        decoder_type: str = "linear",
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_type = decoder_type
        if decoder_type == "linear":
            self.proj = nn.Conv2d(in_dim, num_classes, kernel_size=1)
        elif decoder_type == "conv_lite":
            hidden_dim = hidden_dim or min(128, in_dim)
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive for conv_lite decoder")
            self.proj = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
            )
        else:
            raise ValueError(f"unknown segmentation decoder: {decoder_type!r}")

    def forward(self, x, spatial_dims=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if spatial_dims is None:
                spatial_dims = (1, 1)
        elif x.dim() == 3:
            if spatial_dims is None:
                side = int(math.sqrt(x.shape[1]))
                if side * side != x.shape[1]:
                    raise ValueError(
                        "spatial_dims is required when the token count is not square"
                    )
                spatial_dims = (side, side)
        else:
            raise ValueError(f"expected 2D or 3D features, got shape {tuple(x.shape)}")

        h, w = spatial_dims
        if x.shape[1] != h * w:
            raise ValueError(
                f"spatial_dims {spatial_dims!r} do not match token count {x.shape[1]}"
            )
        bsz = x.shape[0]
        x = x.transpose(1, 2).reshape(bsz, -1, h, w)  # [B, D, h, w]
        x = self.proj(x)  # [B, C, h, w]
        return F.interpolate(
            x,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False,
        )
