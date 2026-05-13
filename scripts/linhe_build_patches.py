"""Build Linhe ROI patch dataset.

Stage 1 (default, no external deps):
  Read each .tif from results/linhe/linhe_scenes.geojson, resample to a target
  GSD (default 10 m to match Sentinel-2 / AlphaEarth grid), tile into
  fixed-size patches (default 128x128), write as compressed npz under
  data/linhe_patches/<scene_id>/patch_<row>_<col>.npz.

Stage 2 (--with-s2, requires earthengine-api + GEE auth):
  For each scene's footprint + date window, pull Sentinel-2 L2A 6-band
  (B2/B3/B4/B8/B11/B12) and write paired patches under
  data/linhe_patches/<scene_id>/s2/.

Outputs:
  data/linhe_patches/_index.parquet  (one row per patch with paths + bbox)
  data/linhe_patches/_manifest.json  (build config snapshot)

Usage:
  python scripts/linhe_build_patches.py --target-gsd 10 --patch 128 --max-scenes 4
  python scripts/linhe_build_patches.py --max-scenes 0 --resume         # full 203-scene rebuild
  python scripts/linhe_build_patches.py --quarters 2025Q1 2025Q2 2025Q3 2025Q4 --resume
  python scripts/linhe_build_patches.py --with-s2  # later, after GEE setup
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "results" / "linhe" / "linhe_scenes.geojson"
OUT_BASE = ROOT / "data" / "linhe_patches"


def reproject_to_grid(src: rasterio.io.DatasetReader, target_gsd_m: float) -> tuple[np.ndarray, dict]:
    """Reproject src to EPSG:3857 at target_gsd_m. Returns (array CHW, profile)."""
    dst_crs = "EPSG:3857"
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_gsd_m
    )
    dst = np.zeros((src.count, height, width), dtype=src.dtypes[0])
    for i in range(src.count):
        reproject(
            source=rasterio.band(src, i + 1),
            destination=dst[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    profile = {"crs": dst_crs, "transform": transform, "width": width, "height": height,
               "count": src.count, "dtype": str(src.dtypes[0])}
    return dst, profile


def tile_array(arr: np.ndarray, patch: int, stride: int) -> list[tuple[int, int, np.ndarray]]:
    """Yield (row, col, patch_array CHW) tiles. Skip patches with >50% zeros."""
    c, h, w = arr.shape
    out = []
    for r in range(0, h - patch + 1, stride):
        for col in range(0, w - patch + 1, stride):
            tile = arr[:, r:r + patch, col:col + patch]
            if (tile == 0).mean() > 0.5:
                continue
            out.append((r, col, tile))
    return out


def patch_bbox(profile: dict, row: int, col: int, patch: int) -> tuple[float, float, float, float]:
    t = profile["transform"]
    minx, maxy = t * (col, row)
    maxx, miny = t * (col + patch, row + patch)
    return float(min(minx, maxx)), float(min(miny, maxy)), float(max(minx, maxx)), float(max(miny, maxy))


def build_offline(args: argparse.Namespace) -> None:
    gdf = gpd.read_file(CATALOG)
    if args.satellites:
        gdf = gdf[gdf["satellite"].isin(args.satellites)]
    if args.quarters:
        gdf = gdf[gdf["quarter"].isin(args.quarters)]
    if args.max_scenes:
        gdf = gdf.head(args.max_scenes)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    idx_path = OUT_BASE / "_index.parquet"
    existing_rows: list[dict] = []
    done_scenes: set[str] = set()
    if args.resume and idx_path.exists():
        try:
            existing = pd.read_parquet(idx_path)
            existing_rows = existing.to_dict("records")
            done_scenes = set(existing["scene_id"].unique())
            print(f"[resume] {len(done_scenes)} scenes already in {idx_path.name}")
        except Exception as e:
            print(f"[resume] cannot read {idx_path.name}: {e} — starting fresh")

    todo = gdf[~gdf["scene_id"].isin(done_scenes)] if done_scenes else gdf
    print(f"[info] {len(todo)}/{len(gdf)} scenes to process → {OUT_BASE}")

    index_rows: list[dict] = list(existing_rows)
    n_new_scenes = 0

    for _, row in todo.iterrows():
        scene_id = row["scene_id"]
        tif = Path(row["tif_path"])
        out_dir = OUT_BASE / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with rasterio.open(tif) as src:
                arr, profile = reproject_to_grid(src, args.target_gsd)
        except Exception as e:
            print(f"[skip] {scene_id}: {e}")
            continue
        tiles = tile_array(arr, args.patch, args.stride or args.patch)
        scene_rows: list[dict] = []
        for r, c, tile in tiles:
            npz_path = out_dir / f"p_{r:05d}_{c:05d}.npz"
            np.savez_compressed(npz_path, rgb=tile)
            bbox = patch_bbox(profile, r, c, args.patch)
            scene_rows.append({
                "scene_id": scene_id,
                "satellite": row["satellite"],
                "quarter": row["quarter"],
                "date": row["date"],
                "patch_path": str(npz_path.relative_to(ROOT)),
                "row": r, "col": c,
                "minx": bbox[0], "miny": bbox[1], "maxx": bbox[2], "maxy": bbox[3],
                "patch_size": args.patch,
                "gsd_m": args.target_gsd,
                "modality": "rgb",
            })
        index_rows.extend(scene_rows)
        n_new_scenes += 1
        print(f"[ok]   {scene_id}: {len(tiles)} patches ({n_new_scenes}/{len(todo)})")

        # Incremental save every N scenes so a crash never wipes progress
        if n_new_scenes % args.save_every == 0:
            pd.DataFrame(index_rows).to_parquet(idx_path)
            print(f"[save] {len(index_rows)} patches → {idx_path.name}")

    if not index_rows:
        print("[warn] no patches produced")
        return

    idx = pd.DataFrame(index_rows)
    idx.to_parquet(idx_path)
    (OUT_BASE / "_manifest.json").write_text(json.dumps({
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "target_gsd_m": args.target_gsd,
        "patch": args.patch,
        "stride": args.stride or args.patch,
        "n_patches": len(idx),
        "n_scenes": int(idx["scene_id"].nunique()),
        "modalities": ["rgb"],
        "quarters": sorted(idx["quarter"].unique().tolist()),
    }, indent=2), encoding="utf-8")
    print(f"\n[done] {len(idx)} patches across {idx['scene_id'].nunique()} scenes → {idx_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-gsd", type=float, default=10.0, help="meters per pixel after resample")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--stride", type=int, default=None, help="default = patch (non-overlapping)")
    p.add_argument("--max-scenes", type=int, default=4, help="limit for quick test; 0 = all")
    p.add_argument("--satellites", nargs="*", help="filter e.g. JKF01 GF600")
    p.add_argument("--quarters", nargs="*", help="filter e.g. 2025Q3 2025Q4")
    p.add_argument("--resume", action="store_true",
                   help="skip scenes already present in _index.parquet")
    p.add_argument("--save-every", type=int, default=5,
                   help="flush _index.parquet every N processed scenes")
    p.add_argument("--with-s2", action="store_true", help="also pull Sentinel-2 (requires GEE)")
    args = p.parse_args()
    if args.max_scenes == 0:
        args.max_scenes = None

    if args.with_s2:
        raise NotImplementedError(
            "Sentinel-2 pairing requires GEE credentials. Run offline build first; "
            "S2 stage will land in scripts/linhe_pull_s2.py."
        )
    build_offline(args)


if __name__ == "__main__":
    main()
