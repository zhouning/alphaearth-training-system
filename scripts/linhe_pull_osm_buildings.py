"""Pull OSM buildings inside the Linhe ROI via Overpass API.

The Linhe ROI is ~184 × 285 km — too large for a single Overpass query, so we
tile the bounding box into ~0.5° × 0.5° cells, query each one, then concatenate
and dedup. Output is a single GeoJSON consumable by linhe_rasterize_buildings.py.

Output:
  results/linhe/osm/linhe_buildings.geojson      union of all OSM buildings in ROI
  results/linhe/osm/linhe_buildings_summary.md   counts + per-scene density
  results/linhe/osm/_query_manifest.json         query bbox tiles + retry log

Usage:
  python scripts/linhe_pull_osm_buildings.py
  python scripts/linhe_pull_osm_buildings.py --tile-deg 0.3 --endpoint overpass-api.de
  python scripts/linhe_pull_osm_buildings.py --resume   # skip tiles already cached

Notes:
  - Overpass instances rate-limit hard. Default endpoint rotates through three.
  - Each tile result is cached as JSON under results/linhe/osm/_tiles/ to support resume.
  - This pulls *anything* tagged building=*, including walls/sheds. Filtering to
    'building != no' covers the realistic noise; further cleanup (area threshold)
    happens at rasterize time.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon, shape

ROOT = Path(__file__).resolve().parents[1]
SCENE_CATALOG = ROOT / "results" / "linhe" / "linhe_scenes.geojson"
OUT_DIR = ROOT / "results" / "linhe" / "osm"
TILE_CACHE = OUT_DIR / "_tiles"

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

QUERY_TPL = """
[out:json][timeout:180];
(
  way["building"]["building"!="no"]({s},{w},{n},{e});
  relation["building"]["building"!="no"]({s},{w},{n},{e});
);
out body;
>;
out skel qt;
"""


def roi_bounds_4326() -> tuple[float, float, float, float]:
    g = gpd.read_file(SCENE_CATALOG).to_crs(4326)
    b = g.union_all().bounds
    # snap outwards to a friendly tile grid (0.01° = ~1 km)
    return (round(b[0] - 0.01, 2), round(b[1] - 0.01, 2),
            round(b[2] + 0.01, 2), round(b[3] + 0.01, 2))


def tile_bbox(bbox: tuple[float, float, float, float], tile_deg: float):
    """Yield (minx, miny, maxx, maxy) sub-bboxes covering bbox."""
    minx, miny, maxx, maxy = bbox
    xs = []
    x = minx
    while x < maxx:
        xs.append((x, min(x + tile_deg, maxx)))
        x += tile_deg
    ys = []
    y = miny
    while y < maxy:
        ys.append((y, min(y + tile_deg, maxy)))
        y += tile_deg
    for (x0, x1), (y0, y1) in product(xs, ys):
        yield x0, y0, x1, y1


def overpass_query(bbox: tuple[float, float, float, float], endpoint_idx: int = 0) -> dict:
    """Run one Overpass query with simple endpoint rotation + exponential retry."""
    minx, miny, maxx, maxy = bbox
    body = QUERY_TPL.format(s=miny, w=minx, n=maxy, e=maxx)

    last_err = None
    for attempt in range(3):
        ep = ENDPOINTS[(endpoint_idx + attempt) % len(ENDPOINTS)]
        try:
            r = requests.post(ep, data={"data": body}, timeout=300)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 504, 502, 503):
                wait = 30 * (2 ** attempt)
                print(f"        rate-limited ({r.status_code}) on {ep}, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
        except Exception as e:
            last_err = e
            time.sleep(10 * (2 ** attempt))
    raise RuntimeError(f"Overpass failed after retries on {bbox}: {last_err}")


def osm_to_polygons(osm: dict) -> list[Polygon]:
    """Convert Overpass JSON 'way' elements to closed Polygons."""
    nodes = {el["id"]: (el["lon"], el["lat"])
             for el in osm.get("elements", []) if el["type"] == "node"}
    polys = []
    for el in osm.get("elements", []):
        if el["type"] != "way":
            continue
        coords = [nodes[n] for n in el.get("nodes", []) if n in nodes]
        if len(coords) < 4 or coords[0] != coords[-1]:
            if len(coords) >= 3:
                coords.append(coords[0])
            else:
                continue
        try:
            polys.append(Polygon(coords))
        except Exception:
            continue
    return polys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile-deg", type=float, default=0.5,
                   help="bbox tile size in degrees (smaller = more requests, less likely to time out)")
    p.add_argument("--resume", action="store_true",
                   help="skip tiles whose JSON is already cached")
    p.add_argument("--endpoint-idx", type=int, default=0,
                   help="primary endpoint index in ENDPOINTS (rotates on retry)")
    p.add_argument("--sleep-between", type=float, default=2.0,
                   help="seconds to wait between successful tile queries (politeness)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TILE_CACHE.mkdir(parents=True, exist_ok=True)

    bbox = roi_bounds_4326()
    print(f"[info] ROI bbox (WGS84): {bbox}")
    tiles = list(tile_bbox(bbox, args.tile_deg))
    print(f"[info] {len(tiles)} tiles at {args.tile_deg}° each")

    all_polys: list[Polygon] = []
    log: list[dict] = []

    for i, t in enumerate(tiles):
        cache = TILE_CACHE / f"tile_{i:03d}.json"
        if args.resume and cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[ok]   tile {i+1}/{len(tiles)} {t} (cached, {len(data.get('elements', []))} elements)")
        else:
            print(f"[info] tile {i+1}/{len(tiles)} {t} querying …")
            t0 = time.time()
            try:
                data = overpass_query(t, endpoint_idx=args.endpoint_idx)
            except Exception as e:
                print(f"[fail] tile {i}: {e}")
                log.append({"tile_idx": i, "bbox": t, "error": str(e)})
                continue
            cache.write_text(json.dumps(data), encoding="utf-8")
            elapsed = time.time() - t0
            print(f"[ok]   tile {i+1}/{len(tiles)} → {len(data.get('elements', []))} elements in {elapsed:.1f}s")
            log.append({"tile_idx": i, "bbox": t, "elements": len(data.get("elements", [])),
                        "elapsed_s": round(elapsed, 1)})
            time.sleep(args.sleep_between)

        polys = osm_to_polygons(data)
        all_polys.extend(polys)

    if not all_polys:
        print("[warn] no buildings parsed")
        return

    gdf = gpd.GeoDataFrame({"source": ["osm"] * len(all_polys)},
                            geometry=all_polys, crs="EPSG:4326")
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    gdf["wkb"] = gdf.geometry.apply(lambda g: g.wkb)
    gdf = gdf.drop_duplicates(subset="wkb").drop(columns="wkb")
    gdf = gdf.reset_index(drop=True)
    print(f"[ok]   {n_before} polygons → {len(gdf)} after dedup")

    out = OUT_DIR / "linhe_buildings.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print(f"[done] {out}")

    g3857 = gdf.to_crs(3857)
    areas = g3857.geometry.area
    summary = [
        "# Linhe OSM Buildings Summary",
        "",
        f"- Built at: `{pd.Timestamp.now().isoformat(timespec='seconds')}`",
        f"- ROI bbox: `{bbox}`",
        f"- Tile grid: {len(tiles)} × {args.tile_deg}°",
        f"- Building polygons: **{len(gdf):,}**",
        f"- Total footprint area: **{areas.sum()/1e6:.2f} km²**",
        f"- Median footprint: **{areas.median():.0f} m²**",
        f"- p95 footprint: **{areas.quantile(0.95):.0f} m²**",
        f"- Largest single footprint: **{areas.max():.0f} m²**",
        "",
        "Use `linhe_rasterize_buildings.py` to project these polygons onto the patch grid.",
    ]
    (OUT_DIR / "linhe_buildings_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT_DIR / "_query_manifest.json").write_text(
        json.dumps({"bbox": bbox, "tile_deg": args.tile_deg, "log": log}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] summary → {OUT_DIR / 'linhe_buildings_summary.md'}")


if __name__ == "__main__":
    main()
