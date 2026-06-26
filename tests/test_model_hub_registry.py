import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))

from app.services.model_hub_registry import RegistryValidationError, load_model_registry

REGISTRY_DATA_PATH = repo_root / "ae_backend" / "app" / "data" / "model_hub_models.json"


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


def test_load_model_registry_rejects_invalid_status(tmp_path: Path):
    registry_path = tmp_path / "model_hub_models.json"
    _write_registry(registry_path, [_model_record("lulc_6class_prithvi_houlsby", status="retired")])

    with pytest.raises(RegistryValidationError, match="status"):
        load_model_registry(registry_path)


def test_load_model_registry_requires_top_level_json_list(tmp_path: Path):
    registry_path = tmp_path / "model_hub_models.json"
    registry_path.write_text(json.dumps({"models": [_model_record("lulc_6class_prithvi_houlsby")]}), encoding="utf-8")

    with pytest.raises(RegistryValidationError, match="JSON list"):
        load_model_registry(registry_path)


def test_load_model_registry_wraps_json_syntax_errors(tmp_path: Path):
    registry_path = tmp_path / "model_hub_models.json"
    registry_path.write_text("{", encoding="utf-8")

    with pytest.raises(RegistryValidationError, match="Invalid JSON"):
        load_model_registry(registry_path)


def test_load_model_registry_requires_checkpoint_path_field(tmp_path: Path):
    registry_path = tmp_path / "model_hub_models.json"
    record = _model_record("lulc_6class_prithvi_houlsby")
    del record["checkpoint_path"]
    _write_registry(registry_path, [record])

    with pytest.raises(RegistryValidationError, match="checkpoint_path"):
        load_model_registry(registry_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_spec", [["shape", "H x W x 3"]]),
        ("output_spec", [["type", "class_mask"]]),
        ("class_schema", "background"),
        ("metrics", [["mIoU", 0.2971]]),
        ("supported_sensors", "RGB"),
        ("example_inputs", "examples/linhe_rgb.png"),
    ],
)
def test_load_model_registry_rejects_invalid_container_field_types(tmp_path: Path, field: str, value: object):
    registry_path = tmp_path / "model_hub_models.json"
    record = _model_record("lulc_6class_prithvi_houlsby")
    record[field] = value
    _write_registry(registry_path, [record])

    with pytest.raises(RegistryValidationError, match=field):
        load_model_registry(registry_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", 123),
        ("display_name", 123),
        ("task_type", 123),
        ("backbone", 123),
        ("adapter", 123),
        ("trained_region", 123),
        ("license", 123),
        ("checkpoint_path", 123),
    ],
)
def test_load_model_registry_rejects_invalid_scalar_field_types(tmp_path: Path, field: str, value: object):
    registry_path = tmp_path / "model_hub_models.json"
    record = _model_record("lulc_6class_prithvi_houlsby")
    record[field] = value
    _write_registry(registry_path, [record])

    with pytest.raises(RegistryValidationError, match=field):
        load_model_registry(registry_path)


def test_committed_model_hub_registry_loads_phase_1_models():
    registry = load_model_registry(REGISTRY_DATA_PATH)

    statuses = {model.model_id: model.status for model in registry.models}
    assert statuses == {
        "lulc_6class_prithvi_houlsby": "ready",
        "building_extraction_prithvi": "planned",
        "road_hardscape_prithvi": "planned",
        "water_flood_prithvi": "planned",
        "semantic_change_prithvi": "demo_only",
    }
