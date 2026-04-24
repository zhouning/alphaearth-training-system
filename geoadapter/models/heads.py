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
    """Linear decoder: reshape patch tokens → 1x1 conv → bilinear upsample."""
    def __init__(self, in_dim: int = 768, num_classes: int = 2, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, x, spatial_dims):
        h, w = spatial_dims
        B = x.shape[0]
        x = x.transpose(1, 2).reshape(B, -1, h, w)   # [B, 768, h, w]
        x = self.proj(x)                               # [B, C, h, w]
        x = F.interpolate(x, scale_factor=self.patch_size, mode="bilinear", align_corners=False)
        return x
