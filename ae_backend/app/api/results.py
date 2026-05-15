import json
import os
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results"
CHANGE_DIR = RESULTS_DIR / "linhe_change"
CHANGE_HEATMAP = CHANGE_DIR / "change_heatmap_2025Q1_vs_2025Q4.geojson"
CHANGE_PAIRS_DIR = CHANGE_DIR / "2025Q1_vs_2025Q4"

SUMMARY = {
    "tasks": {
        "buildings_synth": {
            "title": "1.a 建筑物分割 (合成弱标签)",
            "linear": 0.706,
            "houlsby": 0.723,
            "delta": 0.017,
            "label_type": "synth: mean(RGB) >= 140",
            "n_scenes": 73,
            "n_patches": 35920,
            "note": "弱监督 sanity check, PEFT 增益小因为标签与 RGB 强相关",
        },
        "lulc_6class": {
            "title": "1.b 土地利用分类 (Esri LULC 6 类)",
            "linear": 0.177,
            "houlsby": 0.293,
            "delta": 0.116,
            "delta_relative": 0.655,
            "label_type": "Esri LULC 2021-2023",
            "classes": ["water", "trees", "crops", "built", "rangeland", "bare"],
            "note": "真业务标签, PEFT 相对增益 65% — 核心故事",
        },
        "change_2025": {
            "title": "1.c 季度变化检测 (PCA-RX)",
            "n_pairs": 6769,
            "top_pca": 0.386,
            "quarters": ["2025Q1", "2025Q4"],
            "method": "PCA-RX + RGB diff",
            "note": "像素级配准 patch 对, 网格 IoU=1.0",
        },
    },
    "lulc_class_distribution": {
        "crops": 0.556,
        "rangeland": 0.328,
        "built": 0.096,
        "water": 0.019,
        "trees": 0.001,
    },
}


@router.get("/summary")
def get_summary():
    return SUMMARY


@lru_cache(maxsize=1)
def _load_heatmap() -> dict:
    if not CHANGE_HEATMAP.exists():
        raise HTTPException(status_code=404, detail=f"missing {CHANGE_HEATMAP.name}")
    with open(CHANGE_HEATMAP, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/change/heatmap")
def get_change_heatmap(
    top: int = Query(500, ge=1, le=10000, description="return top-N features by mean_pca_score"),
):
    fc = _load_heatmap()
    features = sorted(
        fc["features"],
        key=lambda f: f["properties"].get("mean_pca_score", 0),
        reverse=True,
    )[:top]
    return {"type": "FeatureCollection", "features": features}


_PAIR_RE = re.compile(r"pair_visual_(\d+_\d+)\.png$")


@router.get("/change/pairs")
def get_change_pairs():
    if not CHANGE_PAIRS_DIR.exists():
        return []
    pair_files = sorted(CHANGE_PAIRS_DIR.glob("pair_visual_*.png"))
    fc = _load_heatmap()
    out = []
    for pf in pair_files:
        m = _PAIR_RE.search(pf.name)
        if not m:
            continue
        patch_id = m.group(1)
        suffix = f"p_{patch_id}.npz"
        candidates = [
            f for f in fc["features"] if suffix in f["properties"].get("patch_a", "")
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda f: f["properties"].get("mean_pca_score", 0))
        lon, lat = best["geometry"]["coordinates"]
        pca_url = f"/results/linhe_change/2025Q1_vs_2025Q4/change_pca_rx_{patch_id}.png"
        pair_url = f"/results/linhe_change/2025Q1_vs_2025Q4/{pf.name}"
        pca_path = CHANGE_PAIRS_DIR / f"change_pca_rx_{patch_id}.png"
        out.append({
            "patch_id": patch_id,
            "pair_url": pair_url,
            "pca_url": pca_url if pca_path.exists() else None,
            "lon": lon,
            "lat": lat,
            "mean_pca_score": best["properties"]["mean_pca_score"],
            "mean_rgb_diff": best["properties"].get("mean_rgb_diff"),
        })
    out.sort(key=lambda x: x["mean_pca_score"], reverse=True)
    return out
