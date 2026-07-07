from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT


DEFAULT_ASSET_REGISTRY_PATH = (
    Path(PROJECT_ROOT) / "ae_backend" / "app" / "data" / "model_hub_assets.json"
)


class ModelAssetRegistryError(ValueError):
    """Raised when model asset metadata is invalid."""


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelAssetRegistryError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelAssetRegistryError(f"{label} must be a non-empty string")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelAssetRegistryError(f"{label} must be a list")
    return value


def _validate_local_paths(container: dict[str, Any], label: str) -> None:
    paths = container.get("local_paths", [])
    _require_list(paths, f"{label}.local_paths")
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ModelAssetRegistryError(f"{label}.local_paths entries must be strings")


def _validate_model(record: Any) -> dict[str, Any]:
    item = _require_mapping(record, "asset model")
    _require_string(item.get("model_id"), "model_id")
    _require_string(item.get("runtime_kind"), "runtime_kind")
    _require_string(item.get("promotion_policy"), "promotion_policy")

    weights = _require_mapping(item.get("weights"), "weights")
    test_data = _require_mapping(item.get("test_data"), "test_data")
    _validate_local_paths(weights, "weights")
    _validate_local_paths(test_data, "test_data")
    return dict(item)


def load_model_asset_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path or DEFAULT_ASSET_REGISTRY_PATH)
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    root = _require_mapping(payload, "asset registry")
    models = [
        _validate_model(record)
        for record in _require_list(root.get("models"), "models")
    ]
    model_ids = [model["model_id"] for model in models]
    if len(model_ids) != len(set(model_ids)):
        raise ModelAssetRegistryError("model_id values must be unique")
    return {"version": root.get("version", 1), "models": models}


def _file_presence(local_paths: list[str], project_root: Path) -> dict[str, Any]:
    files = []
    for ref in local_paths:
        path = (project_root / ref).resolve()
        files.append(
            {
                "path": ref,
                "exists": path.exists(),
                "is_dir": path.is_dir(),
            }
        )
    return {
        "available": bool(files) and all(item["exists"] for item in files),
        "files": files,
    }


def build_asset_presence(
    record: dict[str, Any],
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or PROJECT_ROOT)
    weights = _require_mapping(record.get("weights", {}), "weights")
    test_data = _require_mapping(record.get("test_data", {}), "test_data")
    return {
        "model_id": record["model_id"],
        "runtime_kind": record["runtime_kind"],
        "weights": _file_presence(list(weights.get("local_paths", [])), root),
        "test_data": _file_presence(list(test_data.get("local_paths", [])), root),
    }

