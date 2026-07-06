from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services.model_hub_registry import ModelHubRegistry


PAPER12_RESULTS_DIR = Path(PROJECT_ROOT) / "paper12_results"


def _runtime_modes_for_model(model: dict[str, Any]) -> list[str]:
    modes = set(model.get("package_profile", {}).get("runtime_modes", []))
    default_mode = model.get("input_spec", {}).get("default_demo_input_mode")
    if default_mode:
        modes.add(str(default_mode))
    for mode in model.get("input_spec", {}).get("supported_job_input_modes", []):
        modes.add(str(mode))
    if model.get("status") == "demo_only":
        modes.add("cached_demo")
    if model.get("model_id") == "lulc_6class_prithvi_houlsby":
        modes.add("demo_patch")
    return sorted(modes)


def _workflow_level(model: dict[str, Any], runtime_modes: list[str]) -> str:
    if model["model_id"] == "prithvi_crop_classification_arcgis_style":
        return "contract_demo"
    if model["status"] == "ready" and runtime_modes:
        return "runnable_and_evaluable"
    if model["status"] == "ready":
        return "registered_ready"
    if model["status"] == "demo_only":
        return "demo"
    if model["status"] == "planned":
        return "planned"
    return "not_configured"


def _checkpoint(model: dict[str, Any]) -> dict[str, Any]:
    path = model.get("checkpoint_path")
    return {
        "configured": bool(path),
        "path": path,
    }


def _metric_evidence(model: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = model.get("metrics", {})
    evidence = []
    for key, value in metrics.items():
        if key in {"source", "label_source", "readiness"}:
            continue
        if value is None:
            continue
        evidence.append(
            {
                "kind": "metric",
                "label": key,
                "value": value,
                "source": metrics.get("source", "model_hub_registry"),
            }
        )
    if "label_source" in metrics:
        evidence.append(
            {
                "kind": "label_source",
                "label": "label_source",
                "value": metrics["label_source"],
                "source": "model_hub_registry",
            }
        )
    return evidence


def _limitations(model: dict[str, Any]) -> list[str]:
    package_profile = model.get("package_profile", {})
    applicability = package_profile.get("applicability", {})
    limitations = applicability.get("limitations", [])
    if isinstance(limitations, list):
        return [str(item) for item in limitations]
    return []


def _next_steps(model: dict[str, Any]) -> list[str]:
    package_profile = model.get("package_profile", {})
    model_card = package_profile.get("model_card", {})
    next_step = model_card.get("next_step")
    if model["model_id"] == "prithvi_crop_classification_arcgis_style":
        steps = [
            "Attach a validated Prithvi crop head and HLS preprocessing pipeline.",
            "Run independent crop validation before claiming replacement readiness.",
        ]
        if next_step and next_step not in steps:
            steps.insert(0, str(next_step))
        return steps
    if model["status"] in {"planned", "not_configured"}:
        return ["Configure checkpoint, preprocessing, runtime, and validation metrics."]
    if next_step:
        return [str(next_step)]
    return []


def _arcgis_replacement(model: dict[str, Any], workflow_level: str) -> dict[str, Any]:
    if model["model_id"] == "prithvi_crop_classification_arcgis_style":
        return {
            "status": "not_ready",
            "reason": "No validated crop checkpoint or ArcGIS-compatible HLS preprocessing contract is configured.",
        }
    if workflow_level == "runnable_and_evaluable":
        return {
            "status": "partial",
            "reason": "Runnable inside AlphaEarth System, but not packaged as an ArcGIS .dlpk replacement.",
        }
    if workflow_level in {"demo", "contract_demo"}:
        return {
            "status": "demo_only",
            "reason": "Capability is available as a demo workflow, not a validated ArcGIS replacement.",
        }
    return {
        "status": "not_ready",
        "reason": "Capability is registered but lacks a validated runnable replacement workflow.",
    }


def _capability(model: dict[str, Any]) -> dict[str, Any]:
    runtime_modes = _runtime_modes_for_model(model)
    workflow_level = _workflow_level(model, runtime_modes)
    return {
        "id": model["model_id"],
        "display_name": model["display_name"],
        "task_type": model["task_type"],
        "readiness": model["status"],
        "workflow_level": workflow_level,
        "runtime_modes": runtime_modes,
        "input_spec": model.get("input_spec", {}),
        "output_spec": model.get("output_spec", {}),
        "supported_sensors": model.get("supported_sensors", []),
        "trained_region": model.get("trained_region"),
        "checkpoint": _checkpoint(model),
        "evidence": _metric_evidence(model),
        "limitations": _limitations(model),
        "next_steps": _next_steps(model),
        "arcgis_replacement": _arcgis_replacement(model, workflow_level),
    }


def _evidence_source(path: Path, label: str) -> dict[str, Any]:
    available = path.exists()
    payload = {
        "kind": "paper12_benchmark",
        "label": label,
        "path": f"paper12_results/{path.name}",
        "available": available,
    }
    if not available:
        payload["note"] = f"Missing local evidence file: paper12_results/{path.name}"
    return payload


def _evidence_sources() -> list[dict[str, Any]]:
    return [
        _evidence_source(
            PAPER12_RESULTS_DIR / "eurosat_channel_bridge_summary.json",
            "EuroSAT channel bridge",
        ),
        _evidence_source(
            PAPER12_RESULTS_DIR / "landcoverai_segmentation.json",
            "LandCover.ai segmentation",
        ),
        _evidence_source(
            PAPER12_RESULTS_DIR / "loveda_full_finetune_summary.json",
            "LoveDA full fine-tuning",
        ),
    ]


def build_system_capabilities(registry: ModelHubRegistry) -> dict[str, Any]:
    models = [model.to_dict() for model in registry.models]
    capabilities = [_capability(model) for model in models]

    readiness_counts = dict(Counter(model["status"] for model in models))
    for status in ["ready", "evaluable", "demo_only", "planned", "not_configured"]:
        readiness_counts.setdefault(status, 0)
    readiness_counts["evaluable"] = sum(
        1 for item in capabilities if item["workflow_level"] == "runnable_and_evaluable"
    )

    runnable_models = sum(
        1
        for item in capabilities
        if item["workflow_level"] in {"runnable_and_evaluable", "registered_ready"}
    )
    demo_workflows = sum(
        1 for item in capabilities if item["workflow_level"] in {"demo", "contract_demo"}
    )
    planned_workflows = readiness_counts["planned"] + readiness_counts["not_configured"]

    return {
        "system": "AlphaEarth System",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "readiness_counts": readiness_counts,
        "summary": {
            "runnable_models": runnable_models,
            "demo_workflows": demo_workflows,
            "planned_workflows": planned_workflows,
            "arcgis_replacement_ready": False,
        },
        "capabilities": capabilities,
        "evidence_sources": _evidence_sources(),
        "notes": [
            "Capability status is derived from the local registry, runtime declarations, and committed evidence files.",
            "Paper 12 artifacts are supporting evidence only; system readiness is the primary product surface.",
            "ArcGIS replacement readiness is conservative and does not imply .dlpk compatibility.",
        ],
    }
