import torch


class BandSelector:
    """Select specific band indices from a multi-band tensor."""

    def __init__(self, indices: list[int] | None = None):
        self.indices = indices

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.indices is None:
            return x
        return x[self.indices]


class Normalize:
    """Radiometric normalization for satellite imagery."""

    def __init__(self, method: str = "log1p"):
        self.method = method

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 10000)
        if self.method == "log1p":
            x = torch.log1p(x) / 10.0
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (x - mean) / std
