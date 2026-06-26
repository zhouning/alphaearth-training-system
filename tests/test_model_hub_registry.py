import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))

from app.services.model_hub_registry import RegistryValidationError, load_model_registry


def _model_record(model_id: str, *, task_type: str = "semantic_segmentation", status: str = "ready") -> dict:
    return {
        "model_id": model_id,
        "display_name": model_id.replace("_", " ").title(),
        "task_type": task_type,
        "backbone": "Prithvi",
        "adapter": "Houlsby",
        "checkpoint_path": None,
        "input_spec": {"shape": "H x W x 3"},
        "output_spec": {"type": "class_mask"},
        "class_schema": ["background", "built"],
        "metrics": {"mIoU": 0.2971},
        "trained_region": "Linhe",
        "supported_sensors": ["RGB"],
        "license": "research",
        "status": status,
        "example_inputs": ["examples/linhe_rgb.png"],
    }


def _write_registry(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_load_model_registry_returns_models_and_public_payload(tmp_path: Path):
    registry_path = tmp_path / "model_hub_models.json"
    _write_registry(
        registry_path,
        [
            _model_record("lulc_6class_prithvi_houlsby"),
            _model_record(
                "semantic_change_prithvi",
                task_type="change_detection",
                status="demo_only",
            ),
        ],
    )

    registry = load_model_registry(registry_path)

    assert len(registry.models) == 2
    assert registry.get_model("lulc_6class_prithvi_houlsby").task_type == "semantic_segmentation"
    assert registry.get_model("semantic_change_prithvi").status == "demo_only"
    assert registry.to_public_dict()["models"][0]["model_id"] == "lulc_6class_prithvi_houlsby"


def test_load_model_registry_rejects_duplicate_model_ids(tmp_path: Path):
    registry_path = tmp_path / "model_hub_models.json"
    _write_registry(
        registry_path,
        [
            _model_record("lulc_6class_prithvi_houlsby"),
            _model_record("lulc_6class_prithvi_houlsby"),
        ],
    )

    with pytest.raises(RegistryValidationError, match="Duplicate model_id"):
        load_model_registry(registry_path)


def test_load_model_registry_requires_task_type(tmp_path: Path):
    registry_path = tmp_path / "model_hub_models.json"
    record = _model_record("lulc_6class_prithvi_houlsby")
    del record["task_type"]
    _write_registry(registry_path, [record])

    with pytest.raises(RegistryValidationError, match="task_type"):
        load_model_registry(registry_path)
