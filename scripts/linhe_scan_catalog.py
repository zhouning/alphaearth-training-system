"""Scan D:/image/临河 → unified scene catalog.

Outputs:
  results/linhe/linhe_scenes.geojson  (footprint per scene)
  results/linhe/linhe_scenes.csv      (flat summary)
  results/linhe/linhe_summary.md      (human-readable summary)

Each scene row combines:
  - metadata xlsx (云量/日期/sensor) when available
  - sidecar .shp (per-tif footprint) when available
  - raster header (width/height/bands/dtype/crs/res)
  - derived: gsd_m (approx), file_size_mb, quarter

Robust to mixed Chinese/English column names and missing sidecars.
"""
from __future__ import annotations

import glob
import math
import os
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

ROOT = Path(r"D:/image/临河")
OUT = Path(__file__).resolve().parents[1] / "results" / "linhe"
OUT.mkdir(parents=True, exist_ok=True)


def approx_gsd_m(res_deg: float, lat: float) -> float:
    return float(res_deg) * 111_320.0 * math.cos(math.radians(lat))


def read_tfw(tif_path: Path) -> tuple[float, float, float, float] | None:
    """Return (px, py, ulx, uly) from .tfw/.tfwx sidecar if present."""
    for ext in (".tfwx", ".tfw", ".wld"):
        side = tif_path.with_suffix(ext)
        if side.exists():
            try:
                vals = [float(x.strip()) for x in side.read_text().strip().splitlines()[:6]]
                if len(vals) == 6:
                    return vals[0], vals[3], vals[4], vals[5]
            except Exception:
                pass
    return None


def read_raster_header(tif_path: Path) -> dict:
    with rasterio.open(tif_path) as r:
        b = r.bounds
        res = r.res[0]
        # JKF01-style: embedded transform is pixel-space but true georef is in .tfwx
        tfw = read_tfw(tif_path)
        if tfw and r.crs and r.crs.is_geographic and abs(res - 1.0) < 1e-6:
            px, py, ulx, uly = tfw
            res = abs(px)
            left, top = ulx, uly
            right = left + px * r.width
            bottom = top + py * r.height
            b = rasterio.coords.BoundingBox(min(left, right), min(top, bottom),
                                            max(left, right), max(top, bottom))
        cy = (b.bottom + b.top) / 2
        if r.crs and r.crs.is_geographic:
            gsd = approx_gsd_m(res, cy)
        else:
            gsd = float(res)
        return {
            "width": r.width,
            "height": r.height,
            "bands": r.count,
            "dtype": str(r.dtypes[0]),
            "crs": r.crs.to_string() if r.crs else None,
            "res_native": res,
            "gsd_m": round(gsd, 3),
            "footprint_from_raster": box(*b),
            "bounds": tuple(b),
        }


def load_metadata_xlsx(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx)
    cols = df.columns.tolist()
    # Heuristic: map Chinese column names → canonical
    rename = {}
    for c in cols:
        s = str(c)
        if "folder" in s.lower():
            rename[c] = "folder_name"
        elif "云" in s:
            rename[c] = "cloud_cover"
        elif s in ("传感器", "sensor") or "sensor" in s.lower():
            rename[c] = "sensor"
        elif "时间" in s or "date" in s.lower():
            rename[c] = "date"
        elif "季度" in s:
            rename[c] = "quarter"
    df = df.rename(columns=rename)
    if "folder_name" not in df.columns:
        # try first column
        df = df.rename(columns={cols[0]: "folder_name"})
    return df


def parse_sat_from_name(name: str) -> tuple[str, str]:
    m = re.match(r"([A-Z0-9]+)_(\d+)_", name)
    sat = m.group(1) if m else name.split("_")[0]
    sensor_m = re.search(r"_([A-Z]{3}\d*|NAD|PMS\d*|0PMS\d+)_", name)
    sensor = sensor_m.group(1) if sensor_m else ""
    return sat, sensor


def parse_date_from_name(name: str) -> str | None:
    m = re.search(r"_(\d{8})\d{6}_", name)
    if m:
        d = m.group(1)
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    return None


def read_sidecar_shp(tif_folder: Path) -> gpd.GeoDataFrame | None:
    shps = list(tif_folder.glob("*.shp"))
    if not shps:
        return None
    try:
        g = gpd.read_file(shps[0])
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        else:
            g = g.to_crs("EPSG:4326")
        return g
    except Exception:
        return None


def read_footprint_zip(zp: Path) -> gpd.GeoDataFrame | None:
    try:
        g = gpd.read_file(f"zip://{zp}")
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        else:
            g = g.to_crs("EPSG:4326")
        return g
    except Exception:
        return None


def main() -> None:
    rows: list[dict] = []
    meta_map: dict[str, dict] = {}

    for q_dir in sorted(ROOT.iterdir()):
        if not q_dir.is_dir():
            continue
        quarter = q_dir.name

        # load metadata xlsx(s) in this quarter
        for xlsx in q_dir.glob("*.xlsx"):
            try:
                df = load_metadata_xlsx(xlsx)
                for _, r in df.iterrows():
                    key = str(r.get("folder_name", "")).strip()
                    if not key:
                        continue
                    meta_map[key] = {
                        "cloud_cover": r.get("cloud_cover"),
                        "sensor_meta": r.get("sensor"),
                        "date_meta": str(r.get("date")) if pd.notna(r.get("date")) else None,
                    }
            except Exception as e:
                print(f"[WARN] failed to read {xlsx}: {e}")

        # per-tif scenes (recursive — some quarters nest under e.g. 低精度正射成果/)
        # Each scene dir contains exactly one .tif; scene_id = parent dir name.
        tif_paths = sorted(q_dir.rglob("*.tif"))
        for tif in tif_paths:
            sub = tif.parent
            scene_id = sub.name
            sat, sensor = parse_sat_from_name(scene_id)
            date = parse_date_from_name(scene_id)

            try:
                header = read_raster_header(tif)
            except Exception as e:
                print(f"[WARN] cannot open {tif}: {e}")
                continue

            sidecar = read_sidecar_shp(sub)
            if sidecar is not None and len(sidecar) > 0:
                geom = sidecar.geometry.iloc[0]
                geom_source = "sidecar_shp"
            else:
                geom = header["footprint_from_raster"]
                geom_source = "raster_bbox"

            m = meta_map.get(scene_id, {})
            rows.append({
                "scene_id": scene_id,
                "satellite": sat,
                "sensor": sensor or m.get("sensor_meta"),
                "date": date or m.get("date_meta"),
                "quarter": quarter,
                "cloud_cover": m.get("cloud_cover"),
                "tif_path": str(tif),
                "file_size_mb": round(tif.stat().st_size / 1e6, 1),
                "width": header["width"],
                "height": header["height"],
                "bands": header["bands"],
                "dtype": header["dtype"],
                "crs": header["crs"],
                "gsd_m": header["gsd_m"],
                "geom_source": geom_source,
                "geometry": geom,
            })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf_path = OUT / "linhe_scenes.geojson"
    csv_path = OUT / "linhe_scenes.csv"
    gdf.to_file(gdf_path, driver="GeoJSON")
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # quarters missing rasters but with footprint zips → separate index
    zip_rows: list[dict] = []
    for q_dir in sorted(ROOT.iterdir()):
        if not q_dir.is_dir():
            continue
        for zp in q_dir.glob("*.zip"):
            g = read_footprint_zip(zp)
            if g is None:
                continue
            for _, r in g.iterrows():
                zip_rows.append({
                    "quarter": q_dir.name,
                    "source_zip": zp.name,
                    "attrs": {k: v for k, v in r.items() if k != "geometry"},
                    "geometry": r.geometry,
                })
    if zip_rows:
        zgdf = gpd.GeoDataFrame(zip_rows, geometry="geometry", crs="EPSG:4326")
        zgdf.to_file(OUT / "linhe_footprints_from_zip.geojson", driver="GeoJSON")

    # markdown summary
    by_q = gdf.groupby("quarter").agg(
        n=("scene_id", "count"),
        gb=("file_size_mb", lambda s: round(s.sum() / 1024, 2)),
        sats=("satellite", lambda s: ",".join(sorted(set(s)))),
    )
    by_sat = gdf.groupby("satellite").agg(n=("scene_id", "count"), mean_gsd=("gsd_m", "mean"))

    total_union = gdf.unary_union
    ext = total_union.bounds
    center_lat = (ext[1] + ext[3]) / 2
    center_lng = (ext[0] + ext[2]) / 2

    md = [
        "# 临河影像数据清单",
        "",
        f"- 扫描时间: `{pd.Timestamp.now().isoformat(timespec='seconds')}`",
        f"- 总场景数: **{len(gdf)}**",
        f"- 总文件大小: **{gdf['file_size_mb'].sum()/1024:.1f} GB**",
        f"- 覆盖范围 (EPSG:4326): lon [{ext[0]:.4f}, {ext[2]:.4f}], lat [{ext[1]:.4f}, {ext[3]:.4f}]",
        f"- 中心约: ({center_lat:.4f}, {center_lng:.4f})",
        f"- 总覆盖面积 (union): {gdf.to_crs(3857).unary_union.area/1e6:.1f} km²",
        "",
        "## 按季度",
        "",
        by_q.to_markdown(),
        "",
        "## 按卫星",
        "",
        by_sat.to_markdown(),
        "",
        "## 说明",
        "",
        "- `.tif` 为 3 波段 uint8 RGB 成品影像 (**无 NIR/SWIR/红边**)",
        "- 同一 quarter 内 scene 目录可能嵌套（如 `2020Q1/低精度正射成果/<scene>/<scene>.tif`），脚本递归扫描",
        "- 落图 zip 仅作 footprint 兜底, 已被光栅 bbox 取代; 见 `linhe_footprints_from_zip.geojson`",
        f"- 输出 GeoJSON: `{gdf_path.relative_to(Path(__file__).resolve().parents[1])}`",
        f"- 输出 CSV:     `{csv_path.relative_to(Path(__file__).resolve().parents[1])}`",
    ]
    (OUT / "linhe_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] {len(gdf)} scenes → {gdf_path}")
    print(f"[OK] summary → {OUT / 'linhe_summary.md'}")


if __name__ == "__main__":
    main()
