from __future__ import annotations

import csv
import math
from pathlib import Path

from app.core.config import PROJECT_ROOT


CROP_MODEL_ID = "prithvi_crop_classification_arcgis_style"
CROP_CLASSES = [
    "natural_vegetation",
    "forest",
    "corn",
    "soybeans",
    "wetlands",
    "developed_barren",
    "open_water",
    "winter_wheat",
    "alfalfa",
    "fallow_idle_cropland",
    "cotton",
    "sorghum",
    "other",
]
DEMO_PIXEL_COUNTS = {
    "natural_vegetation": 1200,
    "forest": 900,
    "corn": 6400,
    "soybeans": 1800,
    "wetlands": 700,
    "developed_barren": 740,
    "open_water": 240,
    "winter_wheat": 900,
    "alfalfa": 420,
    "fallow_idle_cropland": 500,
    "cotton": 300,
    "sorghum": 260,
    "other": 360,
}
ARTIFACT_FILES = [
    ("png", "crop_preview.png"),
    ("geojson", "crop_polygons.geojson"),
    ("csv", "crop_summary.csv"),
]
SUMMARY_COLUMNS = {"class", "pixels", "fraction"}


def default_crop_demo_dir() -> Path:
    return Path(PROJECT_ROOT) / "results" / "model_hub" / "prithvi_crop_demo"


def _area_fractions(counts: dict[str, int]) -> dict[str, float]:
    total = max(sum(counts.values()), 1)
    return {class_name: round(count / total, 6) for class_name, count in counts.items()}


def _artifact_manifest(crop_dir: Path) -> tuple[list[dict], bool]:
    artifacts = [
        {"kind": kind, "path": str(crop_dir / filename)}
        for kind, filename in ARTIFACT_FILES
    ]
    all_exist = all(Path(artifact["path"]).exists() for artifact in artifacts)
    return artifacts, all_exist


def _read_crop_summary_csv(summary_csv: Path) -> tuple[dict[str, int], dict[str, float]]:
    with summary_csv.open(encoding="utf-8", newline="") as summary_file:
        reader = csv.DictReader(summary_file)
        if not reader.fieldnames or not SUMMARY_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("missing required columns")

        counts: dict[str, int] = {}
        fractions: dict[str, float] = {}
        for row in reader:
            class_name = (row.get("class") or "").strip()
            pixels = (row.get("pixels") or "").strip()
            fraction = (row.get("fraction") or "").strip()
            if not class_name or not pixels or not fraction:
                raise ValueError("blank class, pixels, or fraction")
            pixel_count = int(pixels)
            area_fraction = float(fraction)
            if pixel_count < 0:
                raise ValueError("negative pixels")
            if not math.isfinite(area_fraction) or not 0 <= area_fraction <= 1:
                raise ValueError("fraction outside [0, 1]")
            counts[class_name] = pixel_count
            fractions[class_name] = area_fraction

    if not counts:
        raise ValueError("no class rows")
    return counts, fractions


def summarize_cached_crop_demo(*, options: dict) -> dict:
    crop_dir = Path(options.get("crop_dir") or default_crop_demo_dir())
    counts = dict(DEMO_PIXEL_COUNTS)
    fractions = _area_fractions(counts)
    logs: list[str] = []

    summary_csv = crop_dir / "crop_summary.csv"
    if summary_csv.exists():
        try:
            counts, fractions = _read_crop_summary_csv(summary_csv)
        except (OSError, ValueError, csv.Error) as exc:
            logs.append(
                f"invalid crop_summary.csv at {summary_csv}; "
                f"using deterministic demo summary ({exc})"
            )

    dominant_class = max(counts, key=counts.get)
    artifacts, artifacts_exist = _artifact_manifest(crop_dir)
    artifact_log = (
        "loaded cached crop demo artifacts"
        if artifacts_exist
        else "returned planned artifact paths"
    )
    logs.append(f"{artifact_log} from {crop_dir}")

    return {
        "result": {
            "task": "crop_classification",
            "model_id": CROP_MODEL_ID,
            "summary": {
                "class_pixel_counts": counts,
                "class_area_fraction": fractions,
                "dominant_class": dominant_class,
                "method": "cached ArcGIS-style Prithvi crop package demo",
            },
            "model_package": {
                "package_type": "arcgis_style_pretrained_imagery_model",
                "family": "prithvi_crop_classification",
                "runtime_mode": "cached_demo",
                "class_schema": list(CROP_CLASSES),
            },
        },
        "artifacts": artifacts,
        "logs": logs,
    }
