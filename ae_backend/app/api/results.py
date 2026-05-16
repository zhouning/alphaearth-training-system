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


# ---------------------------------------------------------------------------
# Demo-time helpers: serve real Linhe LULC training results as model assets
# and as training-monitor curves, so the dashboard does not run the live
# WebSocket mock during a customer demo.
# ---------------------------------------------------------------------------

LINHE_RESULTS = PROJECT_ROOT / "linhe_results" / "linhe_lulc_seg.json"

_METHOD_DISPLAY = {
    "linear_probe": ("Linear Probe", "linear_probe", "linear"),
    "bitfit":       ("BitFit Tuning", "bitfit", "bitfit"),
    "lora_r8":      ("LoRA r=8 (split-QKV)", "lora_r8", "lora"),
    "houlsby":      ("Houlsby Adapter", "houlsby", "houlsby"),
    "geoadapter":   ("GeoAdapter (input-stage)", "geoadapter", "geoadapter"),
}


@lru_cache(maxsize=1)
def _load_linhe_results() -> list[dict]:
    if not LINHE_RESULTS.exists():
        return []
    with open(LINHE_RESULTS, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_by_method() -> dict[str, dict]:
    """Group raw seed records by method into (mean, std, params, n)."""
    rows = _load_linhe_results()
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    out: dict[str, dict] = {}
    for m, recs in by_method.items():
        ious = [r["mIoU"] for r in recs if r.get("mIoU") is not None]
        if not ious:
            continue
        mean = sum(ious) / len(ious)
        var = sum((x - mean) ** 2 for x in ious) / max(len(ious) - 1, 1)
        out[m] = {
            "mean": mean,
            "std": var ** 0.5,
            "params": recs[0].get("trainable_params"),
            "n_seeds": len(ious),
            "ious": sorted(ious, reverse=True),
        }
    return out


@router.get("/models")
def get_models():
    """Return demo-friendly model asset cards from the real Linhe LULC run.

    Each card is a (method × Linhe LULC dataset) tuple with mean mIoU as the
    headline score. Houlsby is auto-marked active. This lets the 模型资产
    tab populate without depending on the AeModel DB table.
    """
    agg = _aggregate_by_method()
    if not agg:
        return []
    cards = []
    method_order = ["houlsby", "geoadapter", "bitfit", "lora_r8", "linear_probe"]
    for method in method_order:
        if method not in agg or method not in _METHOD_DISPLAY:
            continue
        a = agg[method]
        display, _id, _short = _METHOD_DISPLAY[method]
        cards.append({
            "id": f"linhe_lulc_{method}",
            "model_name": f"{display} · Linhe LULC 6-class",
            "evaluation_score": round(a["mean"] * 100, 2),  # scaled to /100 for the existing UI
            "weights_obs_key": f"obs://alphaearth/linhe_lulc/{method}/seeds_42_123_456.pt",
            "is_active": method == "houlsby",
            "created_at": "2026-05-15T16:00:00",
            "dataset_name": "Linhe 35920 patches (Esri 2022 LULC)",
            "training_job_id": f"job_linhe_lulc_{method}",
            "method": method,
            "trainable_params": a["params"],
            "mIoU_mean": round(a["mean"], 4),
            "mIoU_std": round(a["std"], 4),
            "mIoU_per_seed": [round(x, 4) for x in a["ious"]],
            "n_seeds": a["n_seeds"],
        })
    return cards


@router.get("/training_history")
def get_training_history():
    """Return the 5-method × 3-seed Linhe LULC training summary in chart form.

    Used by the 训练监控 tab as static reference curves so demos do not
    rely on a live WebSocket. Returns:
      - methods: ordered list with mean/std mIoU per method
      - per_seed: per-seed scatter for the latent-space chart
    """
    agg = _aggregate_by_method()
    if not agg:
        return {"methods": [], "per_seed": [], "linear_probe_floor": 1 / 6}

    method_order = ["linear_probe", "bitfit", "lora_r8", "geoadapter", "houlsby"]
    methods = []
    per_seed = []
    for m in method_order:
        if m not in agg or m not in _METHOD_DISPLAY:
            continue
        display, _, _ = _METHOD_DISPLAY[m]
        a = agg[m]
        methods.append({
            "key": m,
            "label": display,
            "mean": round(a["mean"], 4),
            "std": round(a["std"], 4),
            "params": a["params"],
            "ious": [round(x, 4) for x in a["ious"]],
        })
        for i, iou in enumerate(a["ious"]):
            per_seed.append({"method": m, "label": display, "seed_idx": i, "mIoU": round(iou, 4)})

    return {
        "task": "Linhe LULC 6-class segmentation (Esri 2022 labels, scene-level split)",
        "methods": methods,
        "per_seed": per_seed,
        "linear_probe_floor": 1 / 6,
        "patches": 35920,
        "scenes": 73,
        "epochs_per_run": 30,
    }
