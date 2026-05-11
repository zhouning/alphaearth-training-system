"""Rasterize OSM building polygons onto the existing Linhe patch grid.

Reads:
  data/linhe_patches/_index.parquet                       (patch bbox in EPSG:3857)
  results/linhe/osm/linhe_buildings.geojson               (WGS84 polygons)

Writes:
  data/linhe_patches/<scene>/osm_buildings_<patchstem>.npz   (binary mask, uint8)
  data/linhe_patches/_osm_index.parquet                       (patch × building stats)
  results/linhe/osm/rasterize_summary.md                      (per-scene coverage report)

Algorithm:
  1. Reproject buildings to EPSG:3857 once.
  2. Build an STR-tree spatial index for O(log N) bbox lookups.
  3. For each patch: query candidates by bbox, clip + rasterize at 10m/pixel.
  4. Save mask + record positive_pixel_count for downstream filtering.

Usage:
  python scripts/linhe_rasterize_buildings.py
  python scripts/linhe_rasterize_buildings.py --min-area-m2 20    # drop sheds/walls
  python scripts/linhe_rasterize_buildings.py --positive-only      # only patches with ≥1 building pixel
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
PATCH_INDEX = ROOT / "data" / "linhe_patches" / "_index.parquet"
BUILDINGS_GEOJSON = ROOT / "results" / "linhe" / "osm" / "linhe_buildings.geojson"
PATCH_BASE = ROOT / "data" / "linhe_patches"
OUT_DIR = ROOT / "results" / "linhe" / "osm"


def load_buildings(min_area_m2: float) -> gpd.GeoDataFrame:
    if not BUILDINGS_GEOJSON.exists():
        raise SystemExit(f"[error] {BUILDINGS_GEOJSON} not found — run linhe_pull_osm_buildings.py first")
    gdf = gpd.read_file(BUILDINGS_GEOJSON).to_crs(3857)
    n0 = len(gdf)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    if min_area_m2 > 0:
        gdf = gdf[gdf.geometry.area >= min_area_m2]
    gdf = gdf.reset_index(drop=True)
    print(f"[info] buildings: {n0} → {len(gdf)} after validity + area≥{min_area_m2} filter")
    return gdf


def build_strtree(gdf: gpd.GeoDataFrame) -> tuple[STRtree, list]:
    geoms = list(gdf.geometry)
    return STRtree(geoms), geoms


def rasterize_patch(patch_row: pd.Series, tree: STRtree, geoms: list) -> np.ndarray:
    """Return uint8 mask (1 = building) at the patch's 3857 bbox + size."""
    minx, miny, maxx, maxy = patch_row["minx"], patch_row["miny"], patch_row["maxx"], patch_row["maxy"]
    size = int(patch_row["patch_size"])
    bbox_geom = box(minx, miny, maxx, maxy)

    idxs = tree.query(bbox_geom)
    if len(idxs) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    hits = [geoms[i] for i in idxs if geoms[i].intersects(bbox_geom)]
    if not hits:
        return np.zeros((size, size), dtype=np.uint8)

    transform = from_bounds(minx, miny, maxx, maxy, size, size)
    mask = features.rasterize(
        ((g, 1) for g in hits),
        out_shape=(size, size),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-area-m2", type=float, default=10.0,
                   help="drop OSM polygons smaller than this (sheds/walls noise)")
    p.add_argument("--positive-only", action="store_true",
                   help="skip patches with zero building pixels (don't write npz, don't index)")
    p.add_argument("--max-patches", type=int, default=0,
                   help="cap patches processed (0 = all) — useful for dry runs")
    args = p.parse_args()

    if not PATCH_INDEX.exists():
        raise SystemExit(f"[error] {PATCH_INDEX} not found — run linhe_build_patches.py first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patches = pd.read_parquet(PATCH_INDEX)
    if args.max_patches:
        patches = patches.head(args.max_patches)
    print(f"[info] patches to process: {len(patches)}")

    buildings = load_buildings(args.min_area_m2)
    if len(buildings) == 0:
        print("[warn] no buildings left after filtering")
        return
    tree, geoms = build_strtree(buildings)
    print(f"[ok]   STR-tree built over {len(geoms)} polygons")

    rows: list[dict] = []
    n_positive, n_written = 0, 0
    for _, r in patches.iterrows():
        mask = rasterize_patch(r, tree, geoms)
        pos = int(mask.sum())
        if pos > 0:
            n_positive += 1

        if args.positive_only and pos == 0:
            continue

        scene_dir = PATCH_BASE / r["scene_id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(r["patch_path"]).stem
        out_npz = scene_dir / f"osm_buildings_{stem}.npz"
        np.savez_compressed(out_npz, mask=mask)
        n_written += 1

        rows.append({
            "scene_id": r["scene_id"],
            "quarter": r["quarter"],
            "date": r["date"],
            "patch_path": r["patch_path"],
            "osm_path": str(out_npz.relative_to(ROOT)),
            "patch_size": int(r["patch_size"]),
            "building_pixels": pos,
            "building_share": round(pos / (int(r["patch_size"]) ** 2), 4),
        })

    if not rows:
        print("[warn] nothing written")
        return

    idx = pd.DataFrame(rows)
    idx_path = PATCH_BASE / "_osm_index.parquet"
    idx.to_parquet(idx_path)
    print(f"[done] {n_written} masks written ({n_positive} contain ≥1 building pixel) → {idx_path}")

    per_scene = idx.groupby("scene_id").agg(
        n_patches=("patch_path", "count"),
        n_with_building=("building_pixels", lambda s: int((s > 0).sum())),
        total_building_px=("building_pixels", "sum"),
        mean_share=("building_share", "mean"),
    ).round(4)

    summary = [
        "# Linhe OSM Building Rasterization Summary",
        "",
        f"- Built at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Buildings used: **{len(buildings):,}** (≥{args.min_area_m2} m²)",
        f"- Patches processed: **{len(patches):,}**",
        f"- Patches with ≥1 building pixel: **{n_positive:,}** ({n_positive/max(len(patches),1)*100:.1f}%)",
        f"- Masks written: **{n_written:,}**",
        "",
        "## Per-scene coverage",
        "",
        per_scene.to_markdown(),
        "",
        "## Notes",
        "",
        "- `building_share` is the fraction of pixels labeled building per patch. Use this",
        "  for class-imbalance-aware sampling at training time.",
        "- Class scheme: `{0: background, 1: building}` — directly consumable by `SegmentationHead(num_classes=2)`.",
    ]
    (OUT_DIR / "rasterize_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT_DIR / "_rasterize_manifest.json").write_text(json.dumps({
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "min_area_m2": args.min_area_m2,
        "positive_only": args.positive_only,
        "n_buildings_used": len(buildings),
        "n_patches_processed": len(patches),
        "n_positive_patches": n_positive,
        "n_masks_written": n_written,
    }, indent=2), encoding="utf-8")
    print(f"[done] summary → {OUT_DIR / 'rasterize_summary.md'}")


if __name__ == "__main__":
    main()
