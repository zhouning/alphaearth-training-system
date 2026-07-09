"""Prepare a Linhe ArcGIS-replacement manual validation packet.

The packet is intentionally not evaluator-ready: it exports RGB chips, Esri
reference masks, previews, and a manifest stub for manual annotation and later
Paper12 inference outputs. It does not create manual truth or model predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from random import Random
from typing import Any, Sequence

import numpy as np
from PIL import Image


PACKET_SCHEMA = "paper12.arcgis_replacement_validation_packet.v1"
CLASS_NAMES = ["background", "built", "crops", "trees", "water", "rangeland_bare"]
PALETTE = np.array(
    [
        [32, 32, 32],
        [204, 83, 75],
        [224, 185, 76],
        [45, 156, 89],
        [64, 137, 201],
        [151, 176, 96],
    ],
    dtype=np.uint8,
)
MANIFEST_COLUMNS = [
    "sample_id",
    "manual_mask_path",
    "manual_label",
    "arcgis_mask_path",
    "arcgis_label",
    "paper12_mask_path",
    "paper12_label",
    "scene_id",
    "x",
    "y",
    "dominant_esri_class",
    "dominant_paper12_class",
    "annotator_id",
    "review_status",
    "manual_mask_target_path",
    "paper12_mask_target_path",
    "source_rgb_path",
    "source_patch_path",
    "source_lulc_path",
    "preview_path",
    "year",
]


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "sample"


def _resolve(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _read_index(index_path: Path) -> list[dict[str, str]]:
    suffix = index_path.suffix.lower()
    if suffix == ".csv":
        with index_path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading parquet indexes requires pandas.") from exc
        try:
            frame = pd.read_parquet(index_path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet indexes requires pyarrow or fastparquet. "
                "Install one of them or export the index to CSV."
            ) from exc
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in frame.to_dict("records")
        ]
    raise ValueError(f"Unsupported index format: {index_path}")


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def discover_lulc_rows_from_patch_root(
    patch_root: str | Path,
    *,
    year: int,
    repo_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Discover Linhe patch/mask pairs without requiring a parquet reader."""
    patch_root = Path(patch_root)
    relative_root = Path(repo_root) if repo_root is not None else patch_root
    rows: list[dict[str, str]] = []
    for patch_path in sorted(patch_root.glob("*/p_*.npz")):
        scene_id = patch_path.parent.name
        lulc_path = patch_path.with_name(f"lulc_{year}_{patch_path.name}")
        if not lulc_path.exists():
            continue
        sample_id = _safe_id(f"{scene_id}_{patch_path.stem}")
        rows.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "year": str(year),
                "patch_path": _relative_or_absolute(patch_path, relative_root),
                "lulc_path": _relative_or_absolute(lulc_path, relative_root),
                "minx": "",
                "miny": "",
                "maxx": "",
                "maxy": "",
                "index_source": "filesystem_scan",
            }
        )
    return rows


def _load_npz_array(path: Path, preferred_key: str) -> np.ndarray:
    payload = np.load(path)
    if preferred_key in payload.files:
        return np.asarray(payload[preferred_key])
    if len(payload.files) == 1:
        return np.asarray(payload[payload.files[0]])
    raise ValueError(f"{path} does not contain {preferred_key!r}: {payload.files}")


def ensure_hwc_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB array in CHW or HWC layout, got {rgb.shape}")
    return rgb.astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(mask, dtype=np.int64), 0, len(PALETTE) - 1)
    return PALETTE[clipped]


def class_fractions(mask: np.ndarray, class_names: Sequence[str] = CLASS_NAMES) -> dict[str, float]:
    flat = np.asarray(mask, dtype=np.int64).ravel()
    total = max(int(flat.size), 1)
    return {
        class_name: float(np.count_nonzero(flat == class_id) / total)
        for class_id, class_name in enumerate(class_names)
    }


def _dominant_class(fractions: dict[str, float]) -> str:
    return max(fractions.items(), key=lambda item: (item[1], item[0]))[0]


def _make_preview(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    rgb_img = Image.fromarray(ensure_hwc_rgb(rgb), mode="RGB")
    mask_img = Image.fromarray(colorize_mask(mask), mode="RGB")
    canvas = Image.new("RGB", (rgb_img.width * 2, rgb_img.height))
    canvas.paste(rgb_img, (0, 0))
    canvas.paste(mask_img, (rgb_img.width, 0))
    return canvas


def _candidate_from_row(
    row: dict[str, str],
    *,
    index_dir: Path,
    row_number: int,
) -> dict[str, Any]:
    patch_path = _resolve(index_dir, row["patch_path"])
    lulc_path = _resolve(index_dir, row["lulc_path"])
    mask = _load_npz_array(lulc_path, "mask")
    fractions = class_fractions(mask)
    sample_id = row.get("sample_id") or f"{row.get('scene_id', 'scene')}_{row_number:04d}"
    return {
        "row": row,
        "row_number": row_number,
        "sample_id": _safe_id(str(sample_id)),
        "patch_path": patch_path,
        "lulc_path": lulc_path,
        "mask": np.asarray(mask, dtype=np.uint8),
        "fractions": fractions,
        "dominant_esri_class": _dominant_class(fractions),
    }


def _filter_rows(
    rows: Sequence[dict[str, str]],
    *,
    year: int | None,
) -> list[dict[str, str]]:
    if year is None:
        return list(rows)
    return [row for row in rows if str(row.get("year", "")).strip() == str(year)]


def select_candidates(
    rows: Sequence[dict[str, str]],
    *,
    index_dir: Path,
    sample_count: int,
    year: int | None = None,
    seed: int = 0,
    required_classes: Sequence[str] = ("water", "crops", "built"),
    min_class_fraction: float = 0.01,
) -> list[dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    filtered_rows = _filter_rows(rows, year=year)
    candidates = [
        _candidate_from_row(row, index_dir=index_dir, row_number=index)
        for index, row in enumerate(filtered_rows)
    ]
    if not candidates:
        raise ValueError("No validation candidates matched the requested filters")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for class_name in required_classes:
        eligible = [
            candidate
            for candidate in candidates
            if candidate["sample_id"] not in selected_ids
            and candidate["fractions"].get(class_name, 0.0) >= min_class_fraction
        ]
        if not eligible:
            continue
        best = max(
            eligible,
            key=lambda candidate: (
                candidate["fractions"].get(class_name, 0.0),
                candidate["sample_id"],
            ),
        )
        selected.append(best)
        selected_ids.add(best["sample_id"])
        if len(selected) >= sample_count:
            return selected

    remaining = [
        candidate for candidate in candidates if candidate["sample_id"] not in selected_ids
    ]
    rng = Random(seed)
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_count - len(selected))])
    return selected[:sample_count]


def _relative_to_output(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _write_manifest(
    *,
    output_dir: Path,
    selected: Sequence[dict[str, Any]],
) -> Path:
    manifest_path = output_dir / "arcgis_replacement_annotation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for candidate in selected:
            row = candidate["row"]
            sample_id = candidate["sample_id"]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "manual_mask_path": "",
                    "manual_label": "",
                    "arcgis_mask_path": f"arcgis_masks/{sample_id}.npy",
                    "arcgis_label": "",
                    "paper12_mask_path": "",
                    "paper12_label": "",
                    "scene_id": row.get("scene_id", ""),
                    "x": row.get("minx", ""),
                    "y": row.get("miny", ""),
                    "dominant_esri_class": candidate["dominant_esri_class"],
                    "dominant_paper12_class": "",
                    "annotator_id": "",
                    "review_status": "pending_manual_annotation",
                    "manual_mask_target_path": f"manual_masks/{sample_id}.npy",
                    "paper12_mask_target_path": f"paper12_masks/{sample_id}.npy",
                    "source_rgb_path": f"rgb/{sample_id}.npy",
                    "source_patch_path": row.get("patch_path", ""),
                    "source_lulc_path": row.get("lulc_path", ""),
                    "preview_path": f"previews/{sample_id}.png",
                    "year": row.get("year", ""),
                }
            )
    return manifest_path


def _write_readme(output_dir: Path, manifest_path: Path) -> None:
    text = f"""# Linhe ArcGIS Replacement Annotation Packet

This packet is not evaluator-ready yet. It contains RGB chips, Esri reference
masks, and preview PNGs for manual review.

1. Annotate each sample independently from the RGB/context imagery.
2. Save manual masks to `manual_masks/<sample_id>.npy` using the Linhe 6-class
   schema: {', '.join(CLASS_NAMES)}.
3. Run Paper12 on the same samples and save outputs to
   `paper12_masks/<sample_id>.npy`.
4. Fill `manual_mask_path` and `paper12_mask_path` in
   `{manifest_path.name}`.
5. Run `scripts/evaluate_arcgis_replacement.py --manifest {manifest_path}`.

Do not change the Paper12 ArcGIS replacement status until the filled manifest
has been evaluated.
"""
    (output_dir / "annotation_readme.md").write_text(text, encoding="utf-8")


def build_validation_packet(
    *,
    index_path: str | Path,
    patch_root: str | Path | None = None,
    output_dir: str | Path,
    sample_count: int = 30,
    year: int | None = 2022,
    seed: int = 0,
    required_classes: Sequence[str] = ("water", "crops", "built"),
    min_class_fraction: float = 0.01,
) -> dict[str, Any]:
    index_value = str(index_path)
    output_dir = Path(output_dir)
    if index_value.lower() == "auto":
        if year is None:
            raise ValueError("year is required when index_path='auto'")
        patch_root_path = Path(patch_root or "data/linhe_patches")
        rows = discover_lulc_rows_from_patch_root(
            patch_root_path,
            year=year,
            repo_root=Path.cwd(),
        )
        index_dir = Path.cwd()
    else:
        index_path = Path(index_path)
        rows = _read_index(index_path)
        index_dir = index_path.parent
        patch_root_path = Path(patch_root) if patch_root is not None else None
    selected = select_candidates(
        rows,
        index_dir=index_dir,
        sample_count=sample_count,
        year=year,
        seed=seed,
        required_classes=required_classes,
        min_class_fraction=min_class_fraction,
    )

    for subdir in ["rgb", "arcgis_masks", "manual_masks", "paper12_masks", "previews"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    for candidate in selected:
        sample_id = candidate["sample_id"]
        rgb = _load_npz_array(candidate["patch_path"], "rgb")
        np.save(output_dir / "rgb" / f"{sample_id}.npy", ensure_hwc_rgb(rgb))
        np.save(output_dir / "arcgis_masks" / f"{sample_id}.npy", candidate["mask"])
        _make_preview(rgb, candidate["mask"]).save(
            output_dir / "previews" / f"{sample_id}.png"
        )

    manifest_path = _write_manifest(output_dir=output_dir, selected=selected)
    _write_readme(output_dir, manifest_path)

    covered = {
        class_name
        for candidate in selected
        for class_name in required_classes
        if candidate["fractions"].get(class_name, 0.0) >= min_class_fraction
    }
    class_order = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    summary: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "index_path": index_value,
        "patch_root": None if patch_root_path is None else str(patch_root_path),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "sample_count": len(selected),
        "year": year,
        "seed": seed,
        "class_names": CLASS_NAMES,
        "critical_classes_requested": list(required_classes),
        "critical_classes_covered": sorted(
            covered, key=lambda class_name: class_order.get(class_name, 999)
        ),
        "manual_ground_truth_available": False,
        "arcgis_reference_available": True,
        "paper12_outputs_available": False,
        "evaluator_ready": False,
        "next_action": (
            "Fill manual_mask_path and paper12_mask_path before running "
            "scripts/evaluate_arcgis_replacement.py."
        ),
        "samples": [
            {
                "sample_id": candidate["sample_id"],
                "scene_id": candidate["row"].get("scene_id", ""),
                "dominant_esri_class": candidate["dominant_esri_class"],
                "arcgis_mask_path": _relative_to_output(
                    output_dir / "arcgis_masks" / f"{candidate['sample_id']}.npy",
                    output_dir,
                ),
                "preview_path": _relative_to_output(
                    output_dir / "previews" / f"{candidate['sample_id']}.png",
                    output_dir,
                ),
            }
            for candidate in selected
        ],
    }
    (output_dir / "packet_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a conservative Linhe manual validation packet."
    )
    parser.add_argument(
        "--index",
        default="auto",
        help="Linhe LULC index CSV/parquet, or 'auto' to scan --patch-root.",
    )
    parser.add_argument(
        "--patch-root",
        default="data/linhe_patches",
        help="Patch root used when --index auto.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper12_results/linhe_arcgis_replacement_validation_packet",
    )
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--required-classes", default="water,crops,built")
    parser.add_argument("--min-class-fraction", type=float, default=0.01)
    args = parser.parse_args(argv)

    summary = build_validation_packet(
        index_path=args.index,
        patch_root=args.patch_root,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        year=args.year,
        seed=args.seed,
        required_classes=_parse_csv_list(args.required_classes),
        min_class_fraction=args.min_class_fraction,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
