from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geoadapter.data.datasets import load_eurosat, load_loveda  # noqa: E402


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def reset_dataset_root(path: Path) -> None:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if resolved == repo or resolved == resolved.anchor:
        raise ValueError(f"Refusing to delete unsafe dataset root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def shape_of(value) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "unknown"
    return tuple(int(dim) for dim in shape)


def scalar_label(value) -> int | float | str:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return str(value)


def unique_values(value, limit: int = 20) -> list[int | float]:
    if hasattr(value, "unique"):
        uniques = value.unique()
        if hasattr(uniques, "detach"):
            uniques = uniques.detach().cpu()
        return [scalar_label(v) for v in uniques[:limit]]
    return []


def inspect_classification_dataset(name: str, ds, max_samples: int) -> None:
    n = len(ds)
    sample_count = min(max_samples, n)
    for idx in range(sample_count):
        image, label = ds[idx]
        print(
            f"[ok] {name} sample={idx} len={n} "
            f"image_shape={shape_of(image)} label={scalar_label(label)}"
        )


def inspect_segmentation_dataset(name: str, ds, max_samples: int) -> None:
    n = len(ds)
    sample_count = min(max_samples, n)
    for idx in range(sample_count):
        image, mask = ds[idx]
        print(
            f"[ok] {name} sample={idx} len={n} image_shape={shape_of(image)} "
            f"mask_shape={shape_of(mask)} mask_values={unique_values(mask)}"
        )


def download_eurosat(root: Path, max_samples: int) -> None:
    for split in ("train", "test"):
        ds = load_eurosat(root=root, modality="s2_full", split=split)
        inspect_classification_dataset(f"EuroSAT {split}", ds, max_samples)


def loveda_splits() -> Iterable[tuple[str, str]]:
    yield "urban", "train"
    yield "rural", "train"
    yield "urban", "val"
    yield "rural", "val"


def download_loveda(root: Path, max_samples: int) -> None:
    for domain, split in loveda_splits():
        ds = load_loveda(root=root, domain=domain, split=split, max_samples=max_samples)
        inspect_segmentation_dataset(f"LoveDA {domain}-{split}", ds, max_samples)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and smoke-test public Paper 12 datasets via existing loaders."
    )
    parser.add_argument(
        "--dataset",
        choices=["eurosat", "loveda", "all"],
        default="all",
        help="Dataset to download and smoke-test.",
    )
    parser.add_argument("--eurosat-root", default="data/eurosat")
    parser.add_argument("--loveda-root", default="data/weights/raw_data/loveda")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Number of samples to inspect per split after download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the selected dataset cache directory before downloading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")

    eurosat_root = resolve_repo_path(args.eurosat_root)
    loveda_root = resolve_repo_path(args.loveda_root)

    if args.force and args.dataset in ("eurosat", "all"):
        reset_dataset_root(eurosat_root)
    if args.force and args.dataset in ("loveda", "all"):
        reset_dataset_root(loveda_root)

    if args.dataset in ("eurosat", "all"):
        download_eurosat(eurosat_root, args.max_samples)
    if args.dataset in ("loveda", "all"):
        download_loveda(loveda_root, args.max_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
