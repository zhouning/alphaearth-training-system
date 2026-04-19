"""Pull Sentinel-2 L2A 6-band data for existing Linhe patches.

Strategy:
  1. Read patch index to get the union bounding box (EPSG:3857 → EPSG:4326)
  2. Download ONE S2 composite from GEE covering the entire scene extent
  3. Tile into 128x128 patches matching the existing spatial grid
  4. Save as s2_<row>_<col>.npz with key 's2' (6 bands: B2,B3,B4,B8,B11,B12)
  5. Update _index.parquet with s2_patch_path column

Usage:
  python scripts/linhe_pull_s2.py
  python scripts/linhe_pull_s2.py --date-window 60  # ±60 days around scene date
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "linhe_patches" / "_index.parquet"
OUT_BASE = ROOT / "data" / "linhe_patches"

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
TARGET_GSD = 10
PATCH_SIZE = 128


def init_gee():
    try:
        ee.Initialize(project=os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0977577668"))
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0977577668"))
    print("[ok] GEE initialized")


def get_scene_bbox_4326(idx: pd.DataFrame) -> tuple[float, float, float, float]:
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon1, lat1 = t.transform(idx["minx"].min(), idx["miny"].min())
    lon2, lat2 = t.transform(idx["maxx"].max(), idx["maxy"].max())
    return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def download_s2_composite(bbox_4326, center_date: str, window_days: int, out_tif: Path):
    from datetime import datetime, timedelta
    dt = datetime.strptime(center_date, "%Y-%m-%d")
    start = (dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

    full_roi = ee.Geometry.Rectangle(list(bbox_4326))
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(full_roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(S2_BANDS)
    )
    count = collection.size().getInfo()
    print(f"[info] S2 images found: {count} ({start} to {end}, cloud<20%)")
    if count == 0:
        raise RuntimeError(f"No S2 images found for {bbox_4326} between {start} and {end}")

    composite = collection.median()

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    lon_min, lat_min, lon_max, lat_max = bbox_4326
    n_cols, n_rows = 8, 6
    lon_step = (lon_max - lon_min) / n_cols
    lat_step = (lat_max - lat_min) / n_rows

    tile_tifs = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            tile_bbox = (
                lon_min + ci * lon_step,
                lat_min + ri * lat_step,
                lon_min + (ci + 1) * lon_step,
                lat_min + (ri + 1) * lat_step,
            )
            tile_roi = ee.Geometry.Rectangle(list(tile_bbox))
            tile_img = composite.clip(tile_roi)

            try:
                url = tile_img.getDownloadURL({
                    "scale": TARGET_GSD,
                    "crs": "EPSG:4326",
                    "region": tile_roi,
                    "format": "GEO_TIFF",
                })
            except Exception as e:
                print(f"[warn] tile ({ri},{ci}) getDownloadURL failed: {e}")
                continue

            tile_path = out_tif.parent / f"s2_tile_{ri}_{ci}.tif"
            session = requests.Session()
            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)

            try:
                with session.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tile_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                tile_tifs.append(tile_path)
                print(f"[ok] tile ({ri},{ci}) downloaded ({tile_path.stat().st_size / 1e6:.1f} MB)")
            except Exception as e:
                print(f"[warn] tile ({ri},{ci}) download failed: {e}")

    if not tile_tifs:
        raise RuntimeError("No S2 tiles downloaded successfully")

    from rasterio.merge import merge
    datasets = [rasterio.open(t) for t in tile_tifs]
    mosaic, mosaic_transform = merge(datasets)
    for ds in datasets:
        ds.close()

    profile = rasterio.open(tile_tifs[0]).profile.copy()
    profile.update(
        height=mosaic.shape[1], width=mosaic.shape[2],
        transform=mosaic_transform, count=mosaic.shape[0],
    )
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(mosaic)

    for t in tile_tifs:
        t.unlink()

    print(f"[ok] S2 mosaic saved: {out_tif} ({out_tif.stat().st_size / 1e6:.1f} MB)")


def reproject_to_3857(src_tif: Path, dst_tif: Path):
    with rasterio.open(src_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds, resolution=TARGET_GSD
        )
        profile = src.profile.copy()
        profile.update(crs="EPSG:3857", transform=transform, width=width, height=height)
        with rasterio.open(dst_tif, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:3857",
                    resampling=Resampling.bilinear,
                )
    print(f"[ok] reprojected to EPSG:3857: {dst_tif}")


def tile_s2_to_patches(s2_tif: Path, idx: pd.DataFrame) -> list[dict]:
    rows = []
    with rasterio.open(s2_tif) as src:
        for _, patch_row in idx.iterrows():
            scene_id = patch_row["scene_id"]
            r, c = int(patch_row["row"]), int(patch_row["col"])
            minx, miny = patch_row["minx"], patch_row["miny"]
            maxx, maxy = patch_row["maxx"], patch_row["maxy"]

            try:
                window = from_bounds(minx, miny, maxx, maxy, src.transform)
                data = src.read(window=window)
            except Exception:
                continue

            if data.shape[1] < PATCH_SIZE or data.shape[2] < PATCH_SIZE:
                padded = np.zeros((len(S2_BANDS), PATCH_SIZE, PATCH_SIZE), dtype=data.dtype)
                padded[:, :data.shape[1], :data.shape[2]] = data[:len(S2_BANDS)]
                data = padded
            else:
                data = data[:len(S2_BANDS), :PATCH_SIZE, :PATCH_SIZE]

            if (data == 0).mean() > 0.5:
                continue

            out_dir = OUT_BASE / scene_id
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / f"s2_{r:05d}_{c:05d}.npz"
            np.savez_compressed(npz_path, s2=data.astype(np.float32))

            rows.append({
                "patch_path": patch_row["patch_path"],
                "s2_patch_path": str(npz_path.relative_to(ROOT)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-window", type=int, default=30)
    args = ap.parse_args()

    idx = pd.read_parquet(INDEX_PATH)
    print(f"[info] {len(idx)} patches, date={idx['date'].iloc[0]}")

    init_gee()

    bbox = get_scene_bbox_4326(idx)
    print(f"[info] scene bbox (4326): {bbox}")

    center_date = str(idx["date"].iloc[0])[:10]

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_tif = Path(tmpdir) / "s2_raw.tif"
        reproj_tif = Path(tmpdir) / "s2_3857.tif"

        download_s2_composite(bbox, center_date, args.date_window, raw_tif)
        reproject_to_3857(raw_tif, reproj_tif)
        s2_rows = tile_s2_to_patches(reproj_tif, idx)

    if not s2_rows:
        print("[warn] no S2 patches produced")
        return

    s2_df = pd.DataFrame(s2_rows)
    merged = idx.merge(s2_df, on="patch_path", how="left")
    merged.to_parquet(INDEX_PATH, index=False)

    n_matched = s2_df["s2_patch_path"].notna().sum()
    print(f"\n[done] {n_matched}/{len(idx)} patches paired with S2 data")
    print(f"[done] _index.parquet updated with s2_patch_path column")


if __name__ == "__main__":
    main()
