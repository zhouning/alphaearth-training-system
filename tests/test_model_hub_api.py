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


def test_model_hub_creates_demo_change_job():
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "semantic_change_prithvi",
            "input_mode": "cached_demo",
            "options": {"output_formats": ["png", "geojson", "csv"]},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "semantic_change_prithvi"
    assert body["status"] in {"pending", "running", "succeeded"}

    loaded = client.get(f"/api/ae/model-hub/jobs/{body['job_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["job_id"] == body["job_id"]


def test_model_hub_rejects_unknown_job_model():
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/ae/model-hub/jobs",
        json={"model_id": "not-a-model", "input_mode": "cached_demo", "options": {}},
    )

    assert response.status_code == 404
    assert "not-a-model" in response.json()["detail"]


def test_model_hub_runs_lulc_demo_patch_job(monkeypatch):
    from app.main import app
    import app.api.model_hub as model_hub_api

    def fake_run_model_hub_job(*, model_id, input_mode, options):
        assert model_id == "lulc_6class_prithvi_houlsby"
        assert input_mode == "demo_patch"
        return {
            "result": {
                "task": "lulc_segmentation",
                "model_id": model_id,
                "summary": {"class_area_fraction": {"crops": 1.0}},
            },
            "artifacts": [{"kind": "json", "path": "inline"}],
            "logs": ["ran fake LULC runtime"],
        }

    monkeypatch.setattr(model_hub_api, "run_model_hub_job", fake_run_model_hub_job)

    client = TestClient(app)
    response = client.post(
        "/api/ae/model-hub/jobs",
        json={"model_id": "lulc_6class_prithvi_houlsby", "input_mode": "demo_patch", "options": {}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["summary"]["class_area_fraction"]["crops"] == 1.0
    assert body["logs"][-1] == "ran fake LULC runtime"


def test_model_hub_runs_cached_change_job(monkeypatch, tmp_path: Path):
    from app.main import app
    import app.services.model_hub_change as change_service

    change_dir = tmp_path / "linhe_change"
    pair_dir = change_dir / "2025Q1_vs_2025Q4"
    pair_dir.mkdir(parents=True)
    (change_dir / "change_heatmap_2025Q1_vs_2025Q4.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    def fake_default_change_dir():
        return change_dir

    monkeypatch.setattr(change_service, "default_change_dir", fake_default_change_dir)

    client = TestClient(app)
    response = client.post(
        "/api/ae/model-hub/jobs",
        json={"model_id": "semantic_change_prithvi", "input_mode": "cached_demo", "options": {"top": 10}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["task"] == "change_detection"
