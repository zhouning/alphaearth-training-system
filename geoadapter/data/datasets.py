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
        "rgb_3band": {"indices": None, "c_in": 3, "name": "Native 3-band RGB"},
        "rgb_sar": {"indices": [3, 2, 1], "c_in": 4, "name": "RGB + SAR VV"},
        "gf2": {"indices": [3, 2, 1, 7], "c_in": 4, "name": "GF-2 (B,G,R,NIR)"},
        "sar_only": {"indices": None, "c_in": 2, "name": "SAR VV+VH"},
        "s2_floods": {"indices": list(range(13)), "c_in": 13, "name": "Sen1Floods11 S2 (13 bands)"},
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


def load_sen1floods11(root: str, modality: str = "s2_floods", split: str = "train",
                      max_samples: int = None):
    """Load Sen1Floods11 for binary flood segmentation via torchgeo."""
    try:
        from torchgeo.datasets import Sen1Floods11
    except ImportError:
        raise ImportError("Sen1Floods11 not available in torchgeo. Use load_landcoverai instead.")

    cfg = ModalityConfig(modality)
    ds = Sen1Floods11(root=root, split=split, download=True)
    ds = _SegmentationDataset(ds, cfg.indices, image_key="image", mask_key="mask")
    if max_samples and len(ds) > max_samples:
        from torch.utils.data import Subset
        import numpy as np
        rng = np.random.RandomState(42)
        indices = rng.choice(len(ds), max_samples, replace=False)
        ds = Subset(ds, indices.tolist())
    return ds


def load_landcoverai(root: str, split: str = "train", max_samples: int = None):
    """Load LandCover.ai for 6-class semantic segmentation via torchgeo."""
    try:
        from torchgeo.datasets import LandCoverAI
    except ImportError:
        raise ImportError("Install torchgeo: pip install geoadapter[bench]")

    ds = LandCoverAI(root=root, split=split, download=True)
    ds = _SegmentationDataset(ds, band_indices=None, image_key="image", mask_key="mask")
    if max_samples and len(ds) > max_samples:
        from torch.utils.data import Subset
        import numpy as np
        rng = np.random.RandomState(42)
        indices = rng.choice(len(ds), max_samples, replace=False)
        ds = Subset(ds, indices.tolist())
    return ds


class _SegmentationDataset(Dataset):
    """Wraps a torchgeo dataset returning (image, mask) for segmentation."""

    def __init__(self, base_dataset, band_indices, image_key="image", mask_key="mask"):
        self.base = base_dataset
        self.indices = band_indices
        self.image_key = image_key
        self.mask_key = mask_key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        img = sample[self.image_key]
        if self.indices is not None:
            img = img[self.indices]
        mask = sample[self.mask_key].long()
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        return img, mask


class _LinhePairedDataset(Dataset):
    """Pair RGB patches (data/linhe_patches/_index.parquet) with rasterized masks.

    Designed for the Linhe demo: 3-band RGB uint8 at /255, and a binary or
    multi-class mask npz produced by linhe_rasterize_buildings.py /
    linhe_pull_esri_lulc.py.
    """

    def __init__(self, paired_rows, root, mask_key="mask"):
        import pandas as pd
        self.rows = paired_rows.reset_index(drop=True)
        self.root = root
        self.mask_key = mask_key

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import numpy as np
        import torch
        r = self.rows.iloc[idx]
        rgb = np.load(self.root / r["patch_path"])["rgb"].astype(np.float32) / 255.0
        mask = np.load(self.root / r["label_path"])[self.mask_key].astype(np.int64)
        return torch.from_numpy(rgb), torch.from_numpy(mask)


def load_linhe_buildings(root: str, split: str = "train", val_frac: float = 0.2,
                         seed: int = 42, max_samples: int = None,
                         positive_min_share: float = 0.0,
                         split_mode: str = "scene"):
    """Load paired Linhe RGB patches + OSM building masks for 2-class segmentation.

    Args:
        root: repo root (the directory containing data/linhe_patches/)
        split: "train" or "val"
        val_frac: fraction held out for validation
        seed: RNG seed for patch-level split and max_samples subsampling
        max_samples: cap patches in the returned split
        positive_min_share: drop patches whose building_share is below this
        split_mode: "scene" (default — deterministic by scene_id, prevents leakage
            across scenes; required for honest generalization numbers) or "patch"
            (random patch-level split within the paired pool; use for smoke tests
            and single-scene debugging where scene-level split is impossible).
    """
    import pandas as pd
    from pathlib import Path

    root = Path(root)
    patch_idx = root / "data" / "linhe_patches" / "_index.parquet"
    osm_idx = root / "data" / "linhe_patches" / "_osm_index.parquet"
    if not patch_idx.exists():
        raise FileNotFoundError(f"{patch_idx} not found — run scripts/linhe_build_patches.py")
    if not osm_idx.exists():
        raise FileNotFoundError(
            f"{osm_idx} not found — run "
            "scripts/linhe_pull_osm_buildings.py then linhe_rasterize_buildings.py"
        )

    patches = pd.read_parquet(patch_idx)
    osm = pd.read_parquet(osm_idx).rename(columns={"osm_path": "label_path"})
    paired = patches.merge(osm[["patch_path", "label_path", "building_share"]],
                           on="patch_path", how="inner")
    if positive_min_share > 0:
        paired = paired[paired["building_share"] >= positive_min_share]

    if split_mode == "scene":
        scenes = sorted(paired["scene_id"].unique())
        n_val = max(1, int(len(scenes) * val_frac))
        val_scenes = set(scenes[-n_val:])
        is_val = paired["scene_id"].isin(val_scenes)
    elif split_mode == "patch":
        shuffled = paired.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        cut = int(len(shuffled) * (1.0 - val_frac))
        paired = shuffled
        is_val = pd.Series([i >= cut for i in range(len(paired))], index=paired.index)
    else:
        raise ValueError(f"split_mode must be 'scene' or 'patch', got {split_mode!r}")

    if split == "val":
        paired = paired[is_val]
    elif split == "train":
        paired = paired[~is_val]
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    if max_samples and len(paired) > max_samples:
        paired = paired.sample(n=max_samples, random_state=seed)

    return _LinhePairedDataset(paired, root, mask_key="mask")


def load_linhe_lulc(root: str, year: int, split: str = "train", val_frac: float = 0.2,
                    seed: int = 42, max_samples: int = None):
    """Load paired Linhe RGB patches + Esri LULC masks for N-class segmentation.

    Year selects which annual mosaic to use (2017-2023 supported by the puller).
    The number of classes comes from the mask file itself (default 6 after the
    Linhe remap, or 9 if --keep-9-classes was passed at pull time).
    """
    import pandas as pd
    from pathlib import Path

    root = Path(root)
    patch_idx = root / "data" / "linhe_patches" / "_index.parquet"
    lulc_idx = root / "data" / "linhe_patches" / "_lulc_index.parquet"
    if not patch_idx.exists():
        raise FileNotFoundError(f"{patch_idx} not found — run scripts/linhe_build_patches.py")
    if not lulc_idx.exists():
        raise FileNotFoundError(f"{lulc_idx} not found — run scripts/linhe_pull_esri_lulc.py")

    patches = pd.read_parquet(patch_idx)
    lulc = pd.read_parquet(lulc_idx)
    lulc = lulc[lulc["year"] == year].rename(columns={"lulc_path": "label_path"})
    if len(lulc) == 0:
        raise ValueError(f"no LULC rows for year={year}; pull it with linhe_pull_esri_lulc.py")
    paired = patches.merge(lulc[["patch_path", "label_path"]], on="patch_path", how="inner")

    scenes = sorted(paired["scene_id"].unique())
    n_val = max(1, int(len(scenes) * val_frac))
    val_scenes = set(scenes[-n_val:])
    if split == "val":
        paired = paired[paired["scene_id"].isin(val_scenes)]
    elif split == "train":
        paired = paired[~paired["scene_id"].isin(val_scenes)]
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    if max_samples and len(paired) > max_samples:
        paired = paired.sample(n=max_samples, random_state=seed)

    return _LinhePairedDataset(paired, root, mask_key="mask")
