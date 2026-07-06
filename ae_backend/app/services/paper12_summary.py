from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services.model_hub_registry import ModelHubRegistry


PAPER12_RESULTS_DIR = Path(PROJECT_ROOT) / "paper12_results"


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _missing_benchmark(benchmark_id: str, label: str, source: str) -> dict[str, Any]:
    return {
        "id": benchmark_id,
        "label": label,
        "metric": None,
        "best_method": None,
        "best_value": None,
        "source": source,
        "status": "missing",
        "note": f"Missing local result file: {source}",
    }


def _best_from_summary(
    *,
    benchmark_id: str,
    label: str,
    source_name: str,
    metric_field: str,
    display_metric: str,
) -> dict[str, Any]:
    path = PAPER12_RESULTS_DIR / source_name
    data = _load_json(path)
    if not isinstance(data, dict):
        return _missing_benchmark(benchmark_id, label, f"paper12_results/{source_name}")

    candidates = []
    for method, values in data.items():
        if isinstance(values, dict) and metric_field in values:
            candidates.append((method, float(values[metric_field])))

    if not candidates:
        return {
            "id": benchmark_id,
            "label": label,
            "metric": display_metric,
            "best_method": None,
            "best_value": None,
            "source": f"paper12_results/{source_name}",
            "status": "missing",
            "note": f"No metric field {metric_field!r} found in {source_name}",
        }

    best_method, best_value = max(candidates, key=lambda item: item[1])
    return {
        "id": benchmark_id,
        "label": label,
        "metric": display_metric,
        "best_method": best_method,
        "best_value": best_value,
        "source": f"paper12_results/{source_name}",
        "status": "available",
    }


def _landcoverai_segmentation() -> dict[str, Any]:
    source_name = "landcoverai_segmentation.json"
    path = PAPER12_RESULTS_DIR / source_name
    rows = _load_json(path)
    if not isinstance(rows, list):
        return _missing_benchmark(
            "landcoverai_segmentation",
            "LandCover.ai semantic segmentation",
            f"paper12_results/{source_name}",
        )

    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if isinstance(row, dict) and "method" in row and "mIoU" in row:
            by_method[str(row["method"])].append(float(row["mIoU"]))

    if not by_method:
        return {
            "id": "landcoverai_segmentation",
            "label": "LandCover.ai semantic segmentation",
            "metric": "mIoU",
            "best_method": None,
            "best_value": None,
            "source": f"paper12_results/{source_name}",
            "status": "missing",
            "note": "No mIoU rows found in landcoverai_segmentation.json",
        }

    means = {method: mean(values) for method, values in by_method.items()}
    best_method = max(means, key=means.get)
    return {
        "id": "landcoverai_segmentation",
        "label": "LandCover.ai semantic segmentation",
        "metric": "mIoU",
        "best_method": best_method,
        "best_value": means[best_method],
        "source": f"paper12_results/{source_name}",
        "status": "available",
    }


def _loveda_full_finetune() -> dict[str, Any]:
    source_name = "loveda_full_finetune_summary.json"
    data = _load_json(PAPER12_RESULTS_DIR / source_name)
    if not isinstance(data, dict):
        return _missing_benchmark(
            "loveda_full_finetune",
            "LoveDA full fine-tuning baseline",
            f"paper12_results/{source_name}",
        )

    candidates = []
    for direction, values in data.items():
        if isinstance(values, dict) and "mIoU_mean" in values:
            candidates.append((str(direction), float(values["mIoU_mean"])))

    if not candidates:
        return {
            "id": "loveda_full_finetune",
            "label": "LoveDA full fine-tuning baseline",
            "metric": "mIoU",
            "best_method": None,
            "best_value": None,
            "source": f"paper12_results/{source_name}",
            "status": "missing",
            "note": "No mIoU_mean values found in loveda_full_finetune_summary.json",
        }

    best_method, best_value = max(candidates, key=lambda item: item[1])
    return {
        "id": "loveda_full_finetune",
        "label": "LoveDA full fine-tuning baseline",
        "metric": "mIoU",
        "best_method": best_method,
        "best_value": best_value,
        "source": f"paper12_results/{source_name}",
        "status": "available",
    }


def _arcgis_replacement_status(model: dict[str, Any]) -> tuple[str, str, str]:
    model_id = model["model_id"]
    status = model["status"]

    if model_id == "prithvi_crop_classification_arcgis_style":
        return (
            "not_yet",
            "No validated crop checkpoint is configured; current runtime is a deterministic contract demo.",
            "Attach a validated Prithvi crop head and HLS preprocessing pipeline.",
        )
    if model_id == "water_flood_prithvi":
        return (
            "not_yet",
            "Flood segmentation is registered as planned and has no configured checkpoint.",
            "Attach a validated flood segmentation checkpoint and 6-band HLS/Sen1Floods11 preprocessing.",
        )
    if status == "ready":
        return (
            "partial",
            "Local model is runnable in AlphaEarth System but is not packaged as an ArcGIS .dlpk.",
            "Validate against target production geographies and package deployment artifacts if ArcGIS compatibility is required.",
        )
    if status == "demo_only":
        return (
            "demo_only",
            "Capability is available only as a cached or deterministic demo.",
            "Replace the demo runtime with a validated checkpoint-backed inference path.",
        )
    return (
        "planned",
        "Capability is registered but not executable in the local model hub.",
        "Configure checkpoint, preprocessing, runtime, and validation metrics.",
    )


def build_paper12_summary(registry: ModelHubRegistry) -> dict[str, Any]:
    models = [model.to_dict() for model in registry.models]
    readiness_counts = dict(Counter(model["status"] for model in models))
    for status in ["ready", "demo_only", "planned", "not_configured"]:
        readiness_counts.setdefault(status, 0)

    benchmarks = [
        _best_from_summary(
            benchmark_id="eurosat_channel_bridge",
            label="EuroSAT channel bridge",
            source_name="eurosat_channel_bridge_summary.json",
            metric_field="overall_accuracy_mean",
            display_metric="overall_accuracy",
        ),
        _best_from_summary(
            benchmark_id="peft_capacity_sweep",
            label="EuroSAT PEFT capacity sweep",
            source_name="peft_capacity_sweep_summary.json",
            metric_field="overall_accuracy_mean",
            display_metric="overall_accuracy",
        ),
        _landcoverai_segmentation(),
        _loveda_full_finetune(),
    ]

    capabilities = []
    for model in models:
        replacement_status, reason, next_step = _arcgis_replacement_status(model)
        capabilities.append(
            {
                "model_id": model["model_id"],
                "display_name": model["display_name"],
                "task_type": model["task_type"],
                "readiness": model["status"],
                "arcgis_replacement_status": replacement_status,
                "reason": reason,
                "next_step": next_step,
                "runtime_modes": model.get("package_profile", {}).get("runtime_modes", []),
            }
        )

    return {
        "paper": "paper12",
        "readiness_counts": readiness_counts,
        "benchmarks": benchmarks,
        "capabilities": capabilities,
        "notes": [
            "Paper 12 results are local benchmark evidence for AlphaEarth System model-hub integration.",
            "ArcGIS replacement status is conservative and does not imply .dlpk compatibility.",
        ],
    }