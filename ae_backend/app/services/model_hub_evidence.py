from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services.model_asset_registry import (
    build_asset_presence,
    load_model_asset_registry,
)
from app.services.model_hub_registry import ModelHubRegistry


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _production_state(
    model: dict[str, Any],
    asset: dict[str, Any] | None,
    presence: dict[str, Any] | None,
) -> str:
    if asset is None or presence is None:
        return "metadata_missing"
    if asset["runtime_kind"] == "training_pipeline":
        return "training_required"
    if not presence["weights"]["available"]:
        return "download_required"
    if not presence["test_data"]["available"]:
        return "test_data_required"
    if model.get("status") == "ready":
        return "production_candidate"
    return "verification_required"


def _missing_presence() -> dict[str, Any]:
    return {"available": False, "files": []}


def _model_evidence(
    *,
    model: dict[str, Any],
    asset: dict[str, Any] | None,
    project_root: Path,
) -> dict[str, Any]:
    presence = build_asset_presence(asset, project_root=project_root) if asset else None
    production_state = _production_state(model, asset, presence)
    runtime_kind = asset["runtime_kind"] if asset else "metadata_missing"
    may_run_real_inference = (
        runtime_kind == "neural_checkpoint"
        and production_state in {"production_candidate", "verification_required"}
        and bool(presence and presence["weights"]["available"])
    )
    weights = asset.get("weights", {}) if asset else {}
    test_data = asset.get("test_data", {}) if asset else {}
    return {
        "model_id": model["model_id"],
        "registry_status": model.get("status"),
        "runtime_kind": runtime_kind,
        "production_state": production_state,
        "may_run_real_inference": may_run_real_inference,
        "weights": {
            **weights,
            "presence": presence["weights"] if presence else _missing_presence(),
        },
        "test_data": {
            **test_data,
            "presence": presence["test_data"] if presence else _missing_presence(),
        },
        "promotion_policy": (
            asset.get("promotion_policy") if asset else "asset_metadata_required"
        ),
    }


def build_model_hub_evidence(
    registry: ModelHubRegistry,
    *,
    asset_registry: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    assets = asset_registry or load_model_asset_registry()
    assets_by_id = {item["model_id"]: item for item in assets["models"]}
    root = Path(project_root or PROJECT_ROOT)
    models = [
        _model_evidence(
            model=model_entry.to_dict(),
            asset=assets_by_id.get(model_entry.model_id),
            project_root=root,
        )
        for model_entry in registry.models
    ]
    return {
        "generated_at": _utc_now(),
        "models": models,
    }
