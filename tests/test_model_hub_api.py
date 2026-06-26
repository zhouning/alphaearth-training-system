import sys
from pathlib import Path

from fastapi.testclient import TestClient


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_model_hub_lists_phase1_models():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/model-hub/models")

    assert response.status_code == 200
    body = response.json()
    model_ids = {model["model_id"] for model in body["models"]}
    assert "lulc_6class_prithvi_houlsby" in model_ids
    assert "semantic_change_prithvi" in model_ids
    assert len(body["models"]) >= 5


def test_model_hub_returns_single_model_details():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/model-hub/models/lulc_6class_prithvi_houlsby")

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "lulc_6class_prithvi_houlsby"
    assert body["task_type"] == "semantic_segmentation"
    assert body["status"] == "ready"


def test_model_hub_returns_404_for_unknown_model():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/model-hub/models/not-a-model")

    assert response.status_code == 404
    assert "not-a-model" in response.json()["detail"]
