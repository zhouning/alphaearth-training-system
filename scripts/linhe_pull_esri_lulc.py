"""Pull Esri Sentinel-2 10m LULC time-series for the Linhe ROI.

Source:
  Impact Observatory / Esri / Microsoft, hosted on GEE as
  ``projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS``
  (annual mosaics 2017--2023, 9 classes, 10 m).

Outputs:
  results/linhe/esri_lulc/<year>.tif                 ROI-clipped 9-class mosaic, EPSG:3857, 10 m
  data/linhe_patches/<scene_id>/lulc_<year>.npz      per-patch mask aligned to existing patches
  data/linhe_patches/_lulc_index.parquet             (patch_path, scene, year, lulc_path, ...)
  results/linhe/esri_lulc/_manifest.json             build config + class map snapshot

Usage:
  python scripts/linhe_pull_esri_lulc.py --years 2021 2022
  python scripts/linhe_pull_esri_lulc.py --years 2021 2022 --skip-patches  # only ROI tif

Requires earthengine-api + a one-time ``earthengine authenticate``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
SCENE_CATALOG = ROOT / "results" / "linhe" / "linhe_scenes.geojson"
PATCH_INDEX = ROOT / "data" / "linhe_patches" / "_index.parquet"
PATCH_BASE = ROOT / "data" / "linhe_patches"
OUT_TIF = ROOT / "results" / "linhe" / "esri_lulc"

EE_COLLECTION = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"

# Esri native 9-class codes (Impact Observatory schema)
ESRI_CLASSES = {
    1: "Water",
    2: "Trees",
    4: "Flooded Vegetation",
    5: "Crops",
    7: "Built Area",
    8: "Bare Ground",
    9: "Snow/Ice",
    10: "Clouds",
    11: "Rangeland",
}

# Linhe-targeted 6-class remap (drop snow/clouds/flooded-veg, fold to local semantics)
#   0: background / nodata
#   1: built area      (建成区)
#   2: crops           (耕地)
#   3: trees           (林地)
#   4: water           (水体)
#   5: rangeland+bare  (草地/裸土, 合并 — 临河戈壁-草原边界本就模糊)
LINHE_6CLASS_MAP = {
    0: 0,   # nodata
    1: 4,   # Water     -> water
    2: 3,   # Trees     -> trees
    4: 4,   # FloodedVeg-> water (临河无显著湿地)
    5: 2,   # Crops     -> crops
    7: 1,   # Built     -> built
    8: 5,   # Bare      -> rangeland+bare
    9: 0,   # Snow/Ice  -> background (临河 ESRI 偶尔误标)
    10: 0,  # Clouds    -> background
    11: 5,  # Rangeland -> rangeland+bare
}
LINHE_6CLASS_NAMES = {
    0: "background",
    1: "built",
    2: "crops",
    3: "trees",
    4: "water",
    5: "rangeland_bare",
}


def linhe_roi_bounds_3857() -> tuple[float, float, float, float]:
    """Union footprint of every scene in the catalog, in EPSG:3857."""
    gdf = gpd.read_file(SCENE_CATALOG).to_crs(3857)
    minx, miny, maxx, maxy = gdf.unary_union.bounds
    # snap to 10 m grid so the mosaic aligns cleanly with patch tiles
    minx = np.floor(minx / 10) * 10
    miny = np.floor(miny / 10) * 10
    maxx = np.ceil(maxx / 10) * 10
    maxy = np.ceil(maxy / 10) * 10
    return float(minx), float(miny), float(maxx), float(maxy)


def export_year_to_tif(year: int, bounds_3857: tuple[float, float, float, float], out_path: Path) -> None:
    """Export one annual mosaic clipped to the ROI as a local GeoTIFF via getDownloadURL."""
    import ee
    import requests
    import tempfile

    ee.Initialize()

    col = ee.ImageCollection(EE_COLLECTION).filterDate(f"{year}-01-01", f"{year}-12-31")
    img = col.mosaic().toUint8().rename("lulc")

    minx, miny, maxx, maxy = bounds_3857
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:3857", evenOdd=False)

    url = img.getDownloadURL({
        "scale": 10,
        "region": region,
        "crs": "EPSG:3857",
        "format": "GEO_TIFF",
    })
    print(f"[info] {year}: downloading from GEE → {out_path.name}")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def remap_to_linhe_6(arr: np.ndarray) -> np.ndarray:
    """Vectorized 9-class → 6-class remap. Unknown codes fall to 0."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for src, dst in LINHE_6CLASS_MAP.items():
        out[arr == src] = dst
    return out


def slice_patches_for_year(year: int, raw_tif: Path, remap_to_6: bool) -> list[dict]:
    """For every existing patch in _index.parquet, read the matching LULC window."""
    if not PATCH_INDEX.exists():
        print(f"[warn] {PATCH_INDEX} not found — run linhe_build_patches.py first")
        return []

    idx = pd.read_parquet(PATCH_INDEX)
    rows: list[dict] = []
    with rasterio.open(raw_tif) as src:
        if src.crs.to_string() != "EPSG:3857":
            raise RuntimeError(f"expected EPSG:3857 LULC tif, got {src.crs}")
        for _, r in idx.iterrows():
            bbox = (r["minx"], r["miny"], r["maxx"], r["maxy"])
            try:
                window = from_bounds(*bbox, transform=src.transform)
                mask = src.read(
                    1,
                    window=window,
                    out_shape=(int(r["patch_size"]), int(r["patch_size"])),
                    resampling=Resampling.nearest,
                ).astype(np.uint8)
            except Exception as e:
                print(f"[skip] {r['patch_path']} year={year}: {e}")
                continue

            if remap_to_6:
                mask = remap_to_linhe_6(mask)

            scene_dir = PATCH_BASE / r["scene_id"]
            scene_dir.mkdir(parents=True, exist_ok=True)
            out_npz = scene_dir / f"lulc_{year}_{Path(r['patch_path']).stem}.npz"
            np.savez_compressed(out_npz, mask=mask)

            rows.append({
                "scene_id": r["scene_id"],
                "quarter": r["quarter"],
                "patch_path": r["patch_path"],
                "lulc_path": str(out_npz.relative_to(ROOT)),
                "year": year,
                "n_classes": 6 if remap_to_6 else 9,
                "class_map": "linhe_6" if remap_to_6 else "esri_9",
                "patch_size": r["patch_size"],
            })
    return rows


def class_histogram(tif: Path) -> dict[int, int]:
    """Quick sanity check: count of each class code in the full ROI."""
    with rasterio.open(tif) as src:
        a = src.read(1)
    vals, counts = np.unique(a, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+", default=[2021, 2022],
                   help="annual mosaics to pull (2017-2023 supported)")
    p.add_argument("--skip-patches", action="store_true",
                   help="only export ROI tif, do not slice into patches")
    p.add_argument("--keep-9-classes", action="store_true",
                   help="store raw 9-class labels instead of remapping to Linhe 6-class")
    p.add_argument("--force", action="store_true", help="re-download even if tif exists")
    args = p.parse_args()

    bounds = linhe_roi_bounds_3857()
    print(f"[info] Linhe ROI bounds (EPSG:3857): {bounds}")
    print(f"[info] ROI area: {(bounds[2]-bounds[0])*(bounds[3]-bounds[1])/1e6:.0f} km²")

    OUT_TIF.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    histograms: dict[str, dict[int, int]] = {}

    for year in args.years:
        if not (2017 <= year <= 2023):
            print(f"[skip] year {year} outside supported 2017-2023")
            continue

        tif_path = OUT_TIF / f"{year}.tif"
        if tif_path.exists() and not args.force:
            print(f"[ok]   {year}.tif already exists, reusing (use --force to overwrite)")
        else:
            try:
                export_year_to_tif(year, bounds, tif_path)
            except ImportError as e:
                print(f"[error] {e}: install earthengine-api + run `earthengine authenticate`")
                return
            except Exception as e:
                print(f"[error] year {year} export failed: {e}")
                continue

        histograms[str(year)] = class_histogram(tif_path)
        print(f"[ok]   {year}: class hist = {histograms[str(year)]}")

        if not args.skip_patches:
            rows = slice_patches_for_year(year, tif_path, remap_to_6=not args.keep_9_classes)
            print(f"[ok]   {year}: sliced {len(rows)} patches")
            all_rows.extend(rows)

    if all_rows:
        idx = pd.DataFrame(all_rows)
        idx_path = PATCH_BASE / "_lulc_index.parquet"
        idx.to_parquet(idx_path)
        print(f"[done] {len(idx)} patch×year masks → {idx_path}")

    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "ee_collection": EE_COLLECTION,
        "years": args.years,
        "roi_bounds_3857": bounds,
        "n_classes": 9 if args.keep_9_classes else 6,
        "class_names": ESRI_CLASSES if args.keep_9_classes else LINHE_6CLASS_NAMES,
        "class_map_9to6": None if args.keep_9_classes else LINHE_6CLASS_MAP,
        "histograms_raw9": histograms,
    }
    (OUT_TIF / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    print(f"[done] manifest → {OUT_TIF / '_manifest.json'}")


if __name__ == "__main__":
    main()
