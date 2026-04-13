import torch
from .base import ModalityAdapter


class ZeroPadAdapter(ModalityAdapter):
    """Baseline: zero-pad or truncate channels to match target."""

    def forward(self, x):
        c_in = x.shape[1]
        if c_in < self.out_channels:
            pad = torch.zeros(
                x.shape[0], self.out_channels - c_in, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=1)
        return x[:, : self.out_channels]
