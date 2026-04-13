import torch.nn as nn


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
    """Simple 1x1 conv segmentation head for per-pixel classification."""
    def __init__(self, in_dim: int = 768, num_classes: int = 7, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_dim, num_classes * patch_size * patch_size)
        self.num_classes = num_classes

    def forward(self, x):
        # x: [B, 768] from CLS token — upsample to spatial
        B = x.shape[0]
        return self.proj(x).view(B, self.num_classes, self.patch_size, self.patch_size)
