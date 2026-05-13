"""Pair Linhe RGB patches across quarters by spatial overlap.

For 1.c (quarterly change detection) we need same-ROI patches across different
quarters. This script consumes data/linhe_patches/_index.parquet and emits a
_pairs.parquet with one row per (patch_a, patch_b) pair from different quarters
whose 10m grid bboxes overlap above an IoU threshold (default 0.8).

Outputs:
  data/linhe_patches/_pairs.parquet — columns:
      patch_path_a, patch_path_b
      scene_id_a, scene_id_b, quarter_a, quarter_b, date_a, date_b
      days_gap, iou, share_a, share_b
  data/linhe_patches/_pairs_summary.md — per quarter-pair counts

Pruning:
  Quarter pairs with whole-ROI IoU < min_quarter_iou (default 0.1) are skipped
  outright; this is read from results/linhe/linhe_quarter_iou.csv.

Usage:
  python scripts/linhe_pair_patches.py
  python scripts/linhe_pair_patches.py --iou-threshold 0.5 --quarters 2025Q1 2025Q2 2025Q3 2025Q4
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "linhe_patches" / "_index.parquet"
QUARTER_IOU = ROOT / "results" / "linhe" / "linhe_quarter_iou.csv"
OUT_DIR = ROOT / "data" / "linhe_patches"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iou-threshold", type=float, default=0.8,
                   help="patch-level IoU lower bound for accepting a pair")
    p.add_argument("--min-quarter-iou", type=float, default=0.1,
                   help="quarter-pair whole-ROI IoU below which we skip patch matching")
    p.add_argument("--quarters", nargs="*",
                   help="restrict to a subset of quarters (default = all)")
    args = p.parse_args()

    if not INDEX.exists():
        raise SystemExit(f"missing {INDEX} — run linhe_build_patches.py first")
    idx = pd.read_parquet(INDEX)
    if args.quarters:
        idx = idx[idx["quarter"].isin(args.quarters)]
        print(f"[info] filter quarters → {sorted(idx['quarter'].unique())}")
    print(f"[info] {len(idx)} patches across {idx['quarter'].nunique()} quarters")

    # quarter prune table
    qiou = None
    if QUARTER_IOU.exists():
        qiou = pd.read_csv(QUARTER_IOU, index_col=0)
        print(f"[info] loaded quarter IoU matrix from {QUARTER_IOU.name}")
    else:
        print(f"[warn] {QUARTER_IOU.name} missing — no quarter pruning")

    # group by quarter and build STRtree per quarter once
    by_q: dict[str, pd.DataFrame] = {q: g.reset_index(drop=True)
                                     for q, g in idx.groupby("quarter")}
    trees: dict[str, tuple[STRtree, list]] = {}
    for q, df in by_q.items():
        geoms = [box(r.minx, r.miny, r.maxx, r.maxy) for r in df.itertuples()]
        trees[q] = (STRtree(geoms), geoms)

    quarters = sorted(by_q.keys())
    pairs: list[dict] = []
    for qa, qb in combinations(quarters, 2):
        if qiou is not None and qa in qiou.index and qb in qiou.columns:
            v = qiou.loc[qa, qb]
            if v < args.min_quarter_iou:
                continue
        df_a = by_q[qa]
        df_b = by_q[qb]
        tree_b, geoms_b = trees[qb]
        n_pair_q = 0
        for r in df_a.itertuples():
            ga = box(r.minx, r.miny, r.maxx, r.maxy)
            cand_idx = tree_b.query(ga)
            for j in cand_idx:
                gb = geoms_b[j]
                inter = ga.intersection(gb).area
                if inter == 0:
                    continue
                union = ga.area + gb.area - inter
                iou = inter / union
                if iou < args.iou_threshold:
                    continue
                rb = df_b.iloc[j]
                pairs.append({
                    "patch_path_a": r.patch_path,
                    "patch_path_b": rb["patch_path"],
                    "scene_id_a": r.scene_id,
                    "scene_id_b": rb["scene_id"],
                    "quarter_a": qa,
                    "quarter_b": qb,
                    "date_a": r.date,
                    "date_b": rb["date"],
                    "days_gap": int((rb["date"] - r.date).days) if pd.notna(rb["date"]) and pd.notna(r.date) else None,
                    "iou": round(float(iou), 4),
                    "share_a": round(float(inter / ga.area), 4),
                    "share_b": round(float(inter / gb.area), 4),
                })
                n_pair_q += 1
        print(f"[ok] {qa} <-> {qb}: {n_pair_q} pairs")

    if not pairs:
        print("[warn] no pairs produced")
        return

    out = pd.DataFrame(pairs)
    out_path = OUT_DIR / "_pairs.parquet"
    out.to_parquet(out_path)
    print(f"\n[done] {len(out)} pairs → {out_path}")

    # markdown summary
    by_qpair = out.groupby(["quarter_a", "quarter_b"]).agg(
        n=("iou", "count"),
        iou_mean=("iou", "mean"),
        days_gap_mean=("days_gap", lambda s: s.dropna().mean() if s.notna().any() else None),
    ).round(2)
    md = [
        "# Linhe 季度时序 patch 配对汇总",
        "",
        f"- 配对阈值: patch IoU ≥ {args.iou_threshold}, quarter ROI IoU ≥ {args.min_quarter_iou}",
        f"- 总配对数: **{len(out)}**",
        f"- 涉及 quarter 对: **{out.groupby(['quarter_a', 'quarter_b']).ngroups}**",
        "",
        "## 按 quarter 对",
        "",
        by_qpair.to_markdown(),
        "",
        "## 按时间间隔",
        "",
        f"- days_gap 中位数: {int(out['days_gap'].dropna().median())}",
        f"- days_gap 范围: {int(out['days_gap'].dropna().min())} ~ {int(out['days_gap'].dropna().max())}",
    ]
    (OUT_DIR / "_pairs_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] summary → {OUT_DIR / '_pairs_summary.md'}")


if __name__ == "__main__":
    main()
