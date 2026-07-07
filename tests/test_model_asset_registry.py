import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_model_asset_registry_loads_public_sources():
    from app.services.model_asset_registry import load_model_asset_registry

    registry = load_model_asset_registry()
    by_model = {item["model_id"]: item for item in registry["models"]}

    crop = by_model["prithvi_crop_classification_arcgis_style"]
    assert crop["runtime_kind"] == "neural_checkpoint"
    assert crop["weights"]["source"] == "huggingface"
    assert "18_band_hls_multitemporal_composite" in crop["test_data"]["input_profile"]

    flood = by_model["water_flood_prithvi"]
    assert flood["runtime_kind"] == "neural_checkpoint"
    assert flood["test_data"]["dataset_id"] == "sen1floods11"

    building = by_model["building_extraction_prithvi"]
    assert building["runtime_kind"] == "training_pipeline"
    assert building["test_data"]["dataset_id"] in {
        "spacenet_buildings",
        "microsoft_building_footprints",
    }


def test_model_asset_registry_reports_local_file_presence(tmp_path: Path):
    from app.services.model_asset_registry import build_asset_presence

    root = tmp_path
    existing = root / "data" / "weights" / "x.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"weights")

    record = {
        "model_id": "x",
        "runtime_kind": "neural_checkpoint",
        "weights": {"local_paths": ["data/weights/x.pt", "data/weights/missing.pt"]},
        "test_data": {"local_paths": ["results/missing.tif"]},
    }

    presence = build_asset_presence(record, project_root=root)

    assert presence["weights"]["available"] is False
    assert presence["weights"]["files"][0]["exists"] is True
    assert presence["weights"]["files"][1]["exists"] is False
    assert presence["test_data"]["available"] is False
