"""Linhe dataset QC report.

Reads results/linhe/linhe_scenes.geojson, augments each scene with quality
signals that cannot be inferred from the filename (read a 512x512 downsampled
thumbnail per tif, estimate black-edge ratio + brightness distribution),
then writes:

  results/linhe/linhe_qc.parquet   — per-scene QC table (re-usable from other scripts)
  results/linhe/linhe_qc_report.md — human-readable markdown with decision tables

Checks:
  - date: re-parse from filename with a wider regex covering new GF2/GF6/ZY1F names
  - gsd_bucket: <0.5m / 0.5-1.5m / >1.5m
  - low_precision_flag: path contains 低精度/低精/预处理
  - black_edge_ratio: fraction of near-zero pixels in a 512x512 thumbnail
  - brightness_mean / brightness_std: on the thumbnail (uint8 space)
  - quarter_iou_matrix: union footprint IoU between all quarter pairs
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "results" / "linhe" / "linhe_scenes.geojson"
OUT_DIR = ROOT / "results" / "linhe"
THUMB = 512
BLACK_THR = 5  # uint8 threshold for "near-zero"

# New-style names embed date as _YYYYMMDD_  (no trailing HHMMSS unlike GF100 old style)
DATE_RE = re.compile(r"_(\d{8})(?:\d{6})?_")


def reparse_date(name: str) -> str | None:
    m = DATE_RE.search(name)
    if not m:
        return None
    d = m.group(1)
    if not (19000000 <= int(d) <= 21000000):
        return None
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"


def gsd_bucket(gsd_m: float) -> str:
    if gsd_m < 0.5:
        return "sub_meter"      # JL1KF02B family ~0.42m
    if gsd_m < 1.5:
        return "sub_1p5m"       # GF2/GF701/JKF01 ~0.5-0.7m
    if gsd_m < 2.5:
        return "around_2m"      # GF1/GF6/GF600/ZY1F ~1.6-2.0m
    return "coarse"             # ZY02C ~4.3m


def read_thumbnail(tif_path: Path) -> np.ndarray | None:
    """Return CHW uint8 thumbnail (THUMB x THUMB) downsampled from full raster."""
    try:
        with rasterio.open(tif_path) as src:
            c = min(3, src.count)
            data = src.read(
                indexes=list(range(1, c + 1)),
                out_shape=(c, THUMB, THUMB),
                resampling=Resampling.average,
            )
        return data.astype(np.uint8, copy=False)
    except Exception:
        return None


def compute_thumb_stats(arr: np.ndarray) -> dict:
    """Black-edge ratio (any-band near-zero pixels) + overall brightness moments.

    Also compute center-only dark ratio on the inner 256x256 crop — rotated
    raster bounding boxes naturally leave ~30% black corners so the full-thumb
    ratio is dominated by geometry; center dark > 0.2 is a real coverage hole.
    """
    any_dark = (arr <= BLACK_THR).all(axis=0)
    black_ratio = float(any_dark.mean())
    # center 256x256 (inner half by side, 25% of area)
    h, w = any_dark.shape
    cy0, cy1 = h // 2 - 128, h // 2 + 128
    cx0, cx1 = w // 2 - 128, w // 2 + 128
    center_dark_ratio = float(any_dark[cy0:cy1, cx0:cx1].mean())
    valid = ~any_dark
    if valid.any():
        vals = arr[:, valid]
        brightness_mean = float(vals.mean())
        brightness_std = float(vals.std())
    else:
        brightness_mean, brightness_std = float("nan"), float("nan")
    return {
        "black_edge_ratio": round(black_ratio, 4),
        "center_dark_ratio": round(center_dark_ratio, 4),
        "brightness_mean": round(brightness_mean, 2) if not np.isnan(brightness_mean) else None,
        "brightness_std": round(brightness_std, 2) if not np.isnan(brightness_std) else None,
    }


def quarter_iou_matrix(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Pairwise IoU of quarter-union footprints in EPSG:3857."""
    qs = sorted(gdf["quarter"].unique())
    metric = gdf.to_crs(3857)
    unions = {q: metric[metric["quarter"] == q].geometry.union_all() for q in qs}
    mat = pd.DataFrame(0.0, index=qs, columns=qs)
    for i, qi in enumerate(qs):
        gi = unions[qi]
        ai = gi.area
        for qj in qs[i:]:
            gj = unions[qj]
            inter = gi.intersection(gj).area
            union = gi.union(gj).area
            iou = (inter / union) if union > 0 else 0.0
            mat.loc[qi, qj] = round(iou, 3)
            mat.loc[qj, qi] = round(iou, 3)
            _ = ai
    return mat


def main() -> None:
    gdf = gpd.read_file(CATALOG)
    print(f"[info] {len(gdf)} scenes from catalog")

    # --- per-scene QC (requires reading each tif once for a 512 thumbnail) ---
    qc_rows = []
    for i, row in gdf.iterrows():
        scene_id = row["scene_id"]
        tif = Path(row["tif_path"])
        date_new = reparse_date(scene_id)
        if date_new is None:
            fallback = row.get("date")
            date_new = str(fallback) if fallback is not None and pd.notna(fallback) else None
        low_prec = any(t in row["tif_path"] for t in ("低精度", "低精", "预处理", "中间"))
        thumb = read_thumbnail(tif)
        if thumb is None:
            stats = {"black_edge_ratio": None, "center_dark_ratio": None,
                     "brightness_mean": None, "brightness_std": None}
            read_ok = False
        else:
            stats = compute_thumb_stats(thumb)
            read_ok = True
        qc_rows.append({
            "scene_id": scene_id,
            "satellite": row["satellite"],
            "quarter": row["quarter"],
            "date": date_new,
            "gsd_m": row["gsd_m"],
            "gsd_bucket": gsd_bucket(row["gsd_m"]),
            "low_precision_flag": low_prec,
            "width": row["width"],
            "height": row["height"],
            "read_ok": read_ok,
            **stats,
            "tif_path": row["tif_path"],
        })
        if (i + 1) % 25 == 0:
            print(f"[info] scanned {i + 1}/{len(gdf)}")
    qc = pd.DataFrame(qc_rows)
    qc_path = OUT_DIR / "linhe_qc.parquet"
    qc.to_parquet(qc_path)
    print(f"[ok] {len(qc)} rows → {qc_path}")

    # --- quarter ROI IoU matrix ---
    iou_mat = quarter_iou_matrix(gdf)
    iou_path = OUT_DIR / "linhe_quarter_iou.csv"
    iou_mat.to_csv(iou_path)
    print(f"[ok] quarter IoU matrix → {iou_path}")

    # --- markdown report ---
    lines: list[str] = []
    L = lines.append

    L("# 临河数据质量体检报告 (QC)")
    L("")
    L(f"- 生成时间: `{pd.Timestamp.now().isoformat(timespec='seconds')}`")
    L(f"- 源 catalog: `{CATALOG.relative_to(ROOT)}` ({len(gdf)} scenes, {gdf['width'].sum() * gdf['height'].sum() // 10**12:.0f} 原像素 T 级)")
    L(f"- 读取失败: **{(~qc['read_ok']).sum()}/{len(qc)}**")
    L("")

    # date recovery
    date_recovered = qc["date"].notna().sum()
    L("## 1. Date 解析")
    L("")
    L(f"- 新解析后有 date: **{date_recovered}/{len(qc)}** (原 catalog 缺 12)")
    miss_date = qc[qc["date"].isna()]
    if len(miss_date):
        L(f"- 仍缺 date: {len(miss_date)} scene")
        L("")
        L(miss_date[["scene_id", "satellite", "quarter"]].to_markdown(index=False))
    else:
        L("- 全部恢复 ✓ (新正则 `_YYYYMMDD(HHMMSS)?_` 覆盖 GF2/GF6/ZY1F 命名)")
    L("")

    # gsd bucket
    L("## 2. GSD 分档")
    L("")
    bucket_tbl = qc.groupby(["gsd_bucket", "satellite"]).size().unstack(fill_value=0)
    L(bucket_tbl.to_markdown())
    L("")
    by_b = qc.groupby("gsd_bucket").agg(n=("scene_id", "count"),
                                         gsd_min=("gsd_m", "min"),
                                         gsd_max=("gsd_m", "max"))
    L("汇总:")
    L("")
    L(by_b.round(3).to_markdown())
    L("")
    L("**含义:** 不同 bucket 原生分辨率相差 4-10×, 强制重采样到 10m (`--target-gsd 10`) 会让 sub_meter 下采 24×, coarse 上采 2.3×。**建议**: 先按 gsd_bucket 分组训练, 最后再做跨 bucket 融合实验。")
    L("")

    # low precision flag
    lp = qc[qc["low_precision_flag"]]
    L("## 3. 低精度标记")
    L("")
    L(f"- 路径含 `低精度/低精/预处理/中间` 的 scene: **{len(lp)}/{len(qc)}**")
    if len(lp):
        by_q = lp.groupby(["quarter", "satellite"]).size().rename("n").reset_index()
        L("")
        L(by_q.to_markdown(index=False))
        L("")
        L("**建议**: 不进监督训练集 (labels 可信度未知), 可留作 unsup/pretrain 数据或仅做可视化。")
    L("")

    # black edge + brightness
    L("## 4. 黑边率 + 亮度 (诊断了旋转 bbox artifact)")
    L("")
    ok = qc[qc["read_ok"]]
    L(f"- 读取成功: {len(ok)}")
    L(f"- `black_edge_ratio`: 全 512×512 缩略图中 RGB 三通道均 ≤ {BLACK_THR} 的像素占比")
    L(f"- `center_dark_ratio`: 内部 256×256 (25% 面积) 同指标, 过滤掉旋转 bbox 的四角黑三角")
    L("")
    L("**关键诊断**: 四角黑约 0.30 是光栅旋转 bbox 的几何 artifact, 不是数据坏; `center_dark_ratio` 才反映真实覆盖空洞。")
    L("")
    by_sat = ok.groupby("satellite").agg(
        n=("scene_id", "count"),
        full_black_mean=("black_edge_ratio", "mean"),
        center_dark_mean=("center_dark_ratio", "mean"),
        center_dark_max=("center_dark_ratio", "max"),
        bright_mean=("brightness_mean", "mean"),
    ).round(3)
    L(by_sat.to_markdown())
    L("")
    severe = ok[ok["center_dark_ratio"] > 0.1].sort_values("center_dark_ratio", ascending=False)
    L(f"- **真覆盖空洞** (`center_dark_ratio > 0.1`): **{len(severe)}** scene")
    if len(severe):
        L("")
        L(severe[["scene_id", "satellite", "quarter", "center_dark_ratio", "black_edge_ratio", "brightness_mean"]].head(30).to_markdown(index=False))
        L("")
        L("**建议**: 这些 scene 的中心都有缺像素, 训练前整张剔除, 否则 patch 里混入大片 0 会污染 loss。")
    else:
        L("_无真覆盖空洞_ ✓")
    L("")

    # quarter IoU
    L("## 5. Quarter ROI 重叠 (IoU 矩阵, EPSG:3857)")
    L("")
    L("只保留 IoU > 0.1 的配对 (IoU=1 意味完全同 ROI, 可直接跨 quarter 做像素级时序):")
    L("")
    pairs = []
    for qi in iou_mat.index:
        for qj in iou_mat.columns:
            if qi >= qj:
                continue
            v = iou_mat.loc[qi, qj]
            if v > 0.1:
                pairs.append({"Q_a": qi, "Q_b": qj, "iou": v})
    pair_df = pd.DataFrame(pairs).sort_values("iou", ascending=False)
    if len(pair_df):
        L(pair_df.head(30).to_markdown(index=False))
    else:
        L("_无 IoU>0.1 的 quarter pair — ROI 漂移严重, 跨 quarter 时序需先做 footprint 裁剪_")
    L("")
    # diag/within-quarter coverage
    self_iou = pd.Series({q: iou_mat.loc[q, q] for q in iou_mat.index}, name="self_iou")
    L("Quarter 自身覆盖 (IoU=1 自检, 非 1 说明 geometry 无效):")
    L("")
    L(self_iou.round(3).to_markdown())
    L("")

    # 2025 seasonal story
    q2025 = [q for q in iou_mat.index if q.startswith("2025")]
    if len(q2025) >= 2:
        sub = iou_mat.loc[q2025, q2025]
        L("### 2025 四季度互 IoU (1.c 季度时序可行性)")
        L("")
        L(sub.round(3).to_markdown())
        L("")

    # recommendations
    L("## 6. 建议训练子集 (供参考, 不等于决策)")
    L("")
    high_q = qc[(~qc["low_precision_flag"]) & (qc["read_ok"]) & (qc["center_dark_ratio"].fillna(1) <= 0.1)]
    by_b_hq = high_q.groupby("gsd_bucket").size().to_dict()
    L(f"- 过滤规则: `not low_precision_flag` AND `read_ok` AND `center_dark_ratio ≤ 0.1`")
    L(f"- 保留 scene: **{len(high_q)}/{len(qc)}**")
    L("")
    L("按 GSD bucket 拆分:")
    L("")
    for b, n in sorted(by_b_hq.items()):
        L(f"- `{b}`: {n} scene")
    L("")
    L("**推荐最小可演示子集** (同源同时段同 ROI, 最适合 1.c 季度时序):")
    L("")
    demo_2025 = high_q[high_q["quarter"].str.startswith("2025")]
    L(f"- 2025Q1-Q4 共 {len(demo_2025)} scene, 卫星 {sorted(demo_2025['satellite'].unique())}, GSD bucket {sorted(demo_2025['gsd_bucket'].unique())}")
    L("")

    md_path = OUT_DIR / "linhe_qc_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] markdown → {md_path}")


if __name__ == "__main__":
    main()
