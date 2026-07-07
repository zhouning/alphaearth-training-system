import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_model_hub_evidence_marks_crop_download_required():
    from app.api.model_hub import get_model_registry
    from app.services.model_hub_evidence import build_model_hub_evidence

    evidence = build_model_hub_evidence(get_model_registry())
    by_model = {item["model_id"]: item for item in evidence["models"]}

    crop = by_model["prithvi_crop_classification_arcgis_style"]
    assert crop["runtime_kind"] == "neural_checkpoint"
    assert crop["production_state"] in {
        "download_required",
        "dependency_required",
        "verification_required",
    }
    assert crop["weights"]["source"] == "huggingface"
    assert crop["may_run_real_inference"] is False


def test_model_hub_evidence_marks_lulc_checkpoint_available():
    from app.api.model_hub import get_model_registry
    from app.services.model_hub_evidence import build_model_hub_evidence

    evidence = build_model_hub_evidence(get_model_registry())
    lulc = {
        item["model_id"]: item
        for item in evidence["models"]
    }["lulc_6class_prithvi_houlsby"]

    assert lulc["runtime_kind"] == "neural_checkpoint"
    assert lulc["weights"]["presence"]["available"] is True
    assert lulc["production_state"] in {
        "verification_required",
        "production_candidate",
    }


def test_system_capabilities_include_production_evidence():
    from app.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/api/ae/system/capabilities")

    assert response.status_code == 200
    body = response.json()
    crop = {
        item["id"]: item
        for item in body["capabilities"]
    }["prithvi_crop_classification_arcgis_style"]

    assert crop["production_evidence"]["runtime_kind"] == "neural_checkpoint"
    assert crop["production_evidence"]["production_state"] == "download_required"


def test_system_verification_reports_production_evidence_checks():
    from app.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/api/ae/system/verification")

    assert response.status_code == 200
    body = response.json()
    checks = [
        check
        for check in body["checks"]
        if check["category"] == "production_evidence"
    ]

    assert checks
    assert any(
        check["capability_id"] == "prithvi_crop_classification_arcgis_style"
        and check["status"] == "warning"
        for check in checks
    )
