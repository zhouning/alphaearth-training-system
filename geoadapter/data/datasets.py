"""Dataset loaders for GeoAdapter benchmarks.

Wraps torchgeo datasets with modality configuration support.
Requires `pip install geoadapter[bench]` for torchgeo dependency.
"""
from torch.utils.data import Dataset


class ModalityConfig:
    """Defines which bands to select and how to label them."""

    PRESETS = {
        "s2_full": {"indices": list(range(10)), "c_in": 10, "name": "Sentinel-2 (B2-B12)"},
        "rgb": {"indices": [3, 2, 1], "c_in": 3, "name": "RGB (B4,B3,B2)"},
        "rgb_sar": {"indices": [3, 2, 1], "c_in": 4, "name": "RGB + SAR VV"},
        "gf2": {"indices": [3, 2, 1, 7], "c_in": 4, "name": "GF-2 (B,G,R,NIR)"},
        "sar_only": {"indices": None, "c_in": 2, "name": "SAR VV+VH"},
    }

    def __init__(self, preset: str):
        cfg = self.PRESETS[preset]
        self.indices = cfg["indices"]
        self.c_in = cfg["c_in"]
        self.name = cfg["name"]


def load_eurosat(root: str, modality: str = "s2_full", split: str = "train"):
    """Load EuroSAT via torchgeo with modality selection."""
    try:
        from torchgeo.datasets import EuroSAT
    except ImportError:
        raise ImportError("Install torchgeo: pip install geoadapter[bench]")

    cfg = ModalityConfig(modality)
    ds = EuroSAT(root=root, split=split, download=True)
    return _BandSubset(ds, cfg.indices, key="image")


def load_bigearthnet(root: str, modality: str = "s2_full", split: str = "train",
                     max_samples: int = None, download: bool = True):
    """Load BigEarthNet-S2 (19-class simplified) via torchgeo with modality selection.

    Args:
        root: Dataset root directory (will download ~32GB compressed on first use).
        modality: Band selection preset from ModalityConfig.
        split: One of 'train', 'val', 'test'.
        max_samples: If set, randomly subsample to this many examples.
        download: Whether to auto-download if missing.
    """
    try:
        from torchgeo.datasets import BigEarthNet
    except ImportError:
        raise ImportError("Install torchgeo: pip install geoadapter[bench]")

    cfg = ModalityConfig(modality)
    ds = BigEarthNet(root=root, split=split, bands="s2", num_classes=19, download=download)
    ds = _BandSubset(ds, cfg.indices, key="image")
    if max_samples and len(ds) > max_samples:
        from torch.utils.data import Subset
        import numpy as np
        rng = np.random.RandomState(42)
        indices = rng.choice(len(ds), max_samples, replace=False)
        ds = Subset(ds, indices.tolist())
    return ds


class _BandSubset(Dataset):
    """Wraps a torchgeo dataset to select specific bands."""

    def __init__(self, base_dataset, band_indices, key="image"):
        self.base = base_dataset
        self.indices = band_indices
        self.key = key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        img = sample[self.key]
        if self.indices is not None:
            img = img[self.indices]
        label = sample.get("label", sample.get("labels", 0))
        if hasattr(label, "float") and label.dim() > 0:
            label = label.float()
        return img, label
