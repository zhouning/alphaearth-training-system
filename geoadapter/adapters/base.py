import torch.nn as nn
from abc import abstractmethod


class ModalityAdapter(nn.Module):
    """Base class for input-stage modality adapters."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x):
        """Map [B, C_in, H, W] -> [B, C_out, H, W]."""
        ...
