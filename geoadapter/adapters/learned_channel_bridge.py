import torch
import torch.nn as nn

from .base import ModalityAdapter


class LearnedChannelBridgeAdapter(ModalityAdapter):
    """Learned 1x1 channel bridge initialized as zero-pad/truncate.

    This isolates the channel-bridge ablation: at epoch 0 it is exactly the
    deterministic baseline, but training can reweight or mix all input bands.
    """

    def __init__(self, in_channels: int, out_channels: int = 6):
        super().__init__(in_channels, out_channels)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.projection.weight.zero_()
            for i in range(min(self.in_channels, self.out_channels)):
                self.projection.weight[i, i, 0, 0] = 1.0

    def forward(self, x):
        return self.projection(x)
