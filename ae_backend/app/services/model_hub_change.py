from __future__ import annotations

import json
from pathlib import Path

from app.core.config import PROJECT_ROOT


def default_change_dir() -> Path:
    return Path(PROJECT_ROOT) / "results" / "linhe_change"


def summarize_cached_linhe_change(*, options: dict) -> dict:
    change_dir = Path(options.get("change_dir") or default_change_dir())
    top = max(int(options.get("top", 50)), 1)
    heatmap = change_dir / "change_heatmap_2025Q1_vs_2025Q4.geojson"
    pair_dir = change_dir / "2025Q1_vs_2025Q4"
    if not heatmap.exists():
        raise FileNotFoundError(f"Missing change heatmap: {heatmap}")

    feature_collection = json.loads(heatmap.read_text(encoding="utf-8"))
    features = feature_collection.get("features", [])
    sorted_features = sorted(
        features,
        key=lambda feature: feature.get("properties", {}).get("mean_pca_score", 0),
        reverse=True,
    )[:top]
    top_score = 0.0
    if sorted_features:
        top_score = float(
            sorted_features[0].get("properties", {}).get("mean_pca_score", 0.0)
        )

    artifacts = [{"kind": "geojson", "path": str(heatmap)}]
    if pair_dir.exists():
        for pair_png in sorted(pair_dir.glob("pair_visual_*.png"))[:top]:
            artifacts.append({"kind": "png", "path": str(pair_png)})

    return {
        "result": {
            "task": "change_detection",
            "model_id": "semantic_change_prithvi",
            "summary": {
                "n_features": len(features),
                "returned_features": len(sorted_features),
                "top_mean_pca_score": top_score,
                "method": "PCA-RX visual change plus semantic differencing slot",
            },
            "features": sorted_features,
        },
        "artifacts": artifacts,
        "logs": [f"loaded cached Linhe change artifacts from {change_dir}"],
    }
