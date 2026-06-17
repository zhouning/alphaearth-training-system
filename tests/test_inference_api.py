import io
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_lulc_endpoint_accepts_png_upload_and_returns_segmentation(monkeypatch):
    from app.main import app

    class StubService:
        def predict_image(self, image: np.ndarray) -> dict:
            assert image.shape == (2, 2, 3)
            return {
                "task": "lulc_segmentation",
                "classes": ["background", "built", "crops", "trees", "water", "rangeland_bare"],
                "model_id": "stub-model",
                "device": "cpu",
                "mask_shape": [2, 2],
                "mask": [[2, 2], [3, 0]],
                "class_pixel_counts": {
                    "background": 1,
                    "built": 1,
                    "crops": 2,
                    "trees": 0,
                    "water": 0,
                    "rangeland_bare": 0,
                },
                "class_area_fraction": {
                    "background": 0.25,
                    "built": 0.25,
                    "crops": 0.5,
                    "trees": 0.0,
                    "water": 0.0,
                    "rangeland_bare": 0.0,
                },
            }

    def fake_service(*, checkpoint_path=None, model_id=None):
        assert checkpoint_path == "demo_segmentation.pt"
        assert model_id == "stub-model"
        return StubService()

    import app.api.inference as inference_api

    monkeypatch.setattr(inference_api, "get_lulc_service", fake_service)

    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    payload.seek(0)

    client = TestClient(app)
    response = client.post(
        "/api/ae/inference/lulc",
        data={"checkpoint_path": "demo_segmentation.pt", "model_id": "stub-model"},
        files={"file": ("patch.png", payload, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "lulc_segmentation"
    assert body["model_id"] == "stub-model"
    assert body["mask_shape"] == [2, 2]
    assert body["class_pixel_counts"]["crops"] == 2


def test_lulc_endpoint_uses_default_segmentation_checkpoint(monkeypatch):
    from app.main import app

    class StubService:
        def predict_image(self, image: np.ndarray) -> dict:
            return {
                "task": "lulc_segmentation",
                "classes": ["background", "built", "crops", "trees", "water", "rangeland_bare"],
                "model_id": "linhe-lulc-geoadapter-seed123",
                "device": "cpu",
                "mask_shape": [2, 2],
                "mask": [[2, 2], [2, 2]],
                "class_pixel_counts": {
                    "background": 0,
                    "built": 0,
                    "crops": 4,
                    "trees": 0,
                    "water": 0,
                    "rangeland_bare": 0,
                },
                "class_area_fraction": {
                    "background": 0.0,
                    "built": 0.0,
                    "crops": 1.0,
                    "trees": 0.0,
                    "water": 0.0,
                    "rangeland_bare": 0.0,
                },
            }

    def fake_service(*, checkpoint_path=None, model_id=None):
        assert checkpoint_path is None
        assert model_id is None
        return StubService()

    import app.api.inference as inference_api

    monkeypatch.setattr(inference_api, "get_lulc_service", fake_service)

    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    payload.seek(0)

    client = TestClient(app)
    response = client.post(
        "/api/ae/inference/lulc",
        files={"file": ("patch.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == "linhe-lulc-geoadapter-seed123"


def test_default_lulc_checkpoint_resolves_to_packaged_geoadapter_model():
    import app.api.inference as inference_api

    checkpoint = inference_api._resolve_checkpoint_path(None)

    assert checkpoint.name == "geoadapter__rgb_3band__seed123.pt"
    assert checkpoint.parent.name == "linhe_lulc"


def test_get_lulc_service_caches_loaded_checkpoint(monkeypatch, tmp_path: Path):
    import app.api.inference as inference_api

    calls = []

    class StubService:
        pass

    def fake_from_checkpoint(**kwargs):
        calls.append(kwargs)
        return StubService()

    monkeypatch.setattr(
        inference_api.LULCInferenceService,
        "from_checkpoint",
        fake_from_checkpoint,
    )

    checkpoint_path = tmp_path / "demo.pt"
    checkpoint_path.write_bytes(b"stub")

    first = inference_api.get_lulc_service(
        checkpoint_path=str(checkpoint_path),
        model_id="cached-model",
    )
    second = inference_api.get_lulc_service(
        checkpoint_path=str(checkpoint_path),
        model_id="cached-model",
    )

    assert first is second
    assert len(calls) == 1


def test_lulc_modes_endpoint_returns_public_and_local_capabilities(monkeypatch):
    from app.main import app
    import app.api.inference as inference_api

    monkeypatch.setattr(
        inference_api,
        "build_lulc_capability_registry",
        lambda: {
            "task": "lulc_segmentation",
            "modes": [{"id": "public_product"}, {"id": "local_model"}],
            "local_models": [{"id": "linhe-lulc-geoadapter-seed123", "ready": True}],
            "public_products": [{"id": "esri_lulc_cache", "status": "ready"}],
            "default_local_model_id": "linhe-lulc-geoadapter-seed123",
        },
    )

    client = TestClient(app)
    response = client.get("/api/ae/inference/lulc/modes")

    assert response.status_code == 200
    body = response.json()
    assert body["default_local_model_id"] == "linhe-lulc-geoadapter-seed123"
    assert {mode["id"] for mode in body["modes"]} == {"public_product", "local_model"}


def test_public_lulc_endpoint_queries_public_provider_by_bbox(monkeypatch):
    from app.main import app
    import app.api.inference as inference_api

    def fake_query_public_lulc(**kwargs):
        assert kwargs == {
            "provider_id": "esri_lulc_cache",
            "year": 2022,
            "bbox": (0.0, 0.0, 40.0, 40.0),
            "bbox_crs": "EPSG:3857",
        }
        return {
            "task": "lulc_segmentation",
            "mode": "public_product",
            "provider_id": "esri_lulc_cache",
            "year": 2022,
            "mask_shape": [2, 2],
            "mask": [[2, 2], [1, 4]],
            "class_pixel_counts": {"crops": 2, "built": 1, "water": 1},
            "class_area_fraction": {"crops": 0.5, "built": 0.25, "water": 0.25},
        }

    monkeypatch.setattr(inference_api, "query_public_lulc", fake_query_public_lulc)

    client = TestClient(app)
    response = client.get(
        "/api/ae/inference/lulc/public",
        params={
            "provider_id": "esri_lulc_cache",
            "year": 2022,
            "minx": 0,
            "miny": 0,
            "maxx": 40,
            "maxy": 40,
            "bbox_crs": "EPSG:3857",
        },
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "public_product"
    assert response.json()["provider_id"] == "esri_lulc_cache"


def test_lulc_evaluate_endpoint_returns_prediction_and_metrics(monkeypatch):
    from app.main import app

    class StubService:
        def predict_image(self, image: np.ndarray) -> dict:
            return {
                "task": "lulc_segmentation",
                "mode": "local_model",
                "classes": ["background", "built", "crops", "trees", "water", "rangeland_bare"],
                "model_id": "eval-model",
                "device": "cpu",
                "mask_shape": [2, 2],
                "mask": [[2, 2], [3, 0]],
                "class_pixel_counts": {},
                "class_area_fraction": {},
            }

    def fake_service(*, checkpoint_path=None, model_id=None):
        return StubService()

    import app.api.inference as inference_api

    monkeypatch.setattr(inference_api, "get_lulc_service", fake_service)

    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB")
    image_payload = io.BytesIO()
    image.save(image_payload, format="PNG")
    image_payload.seek(0)

    label = Image.fromarray(np.array([[2, 2], [3, 3]], dtype=np.uint8), mode="L")
    label_payload = io.BytesIO()
    label.save(label_payload, format="PNG")
    label_payload.seek(0)

    client = TestClient(app)
    response = client.post(
        "/api/ae/inference/lulc/evaluate",
        files={
            "file": ("patch.png", image_payload, "image/png"),
            "label_file": ("label.png", label_payload, "image/png"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["model_id"] == "eval-model"
    assert body["evaluation"]["pixel_accuracy"] == 0.75
    assert body["evaluation"]["per_class_iou"]["crops"] == 1.0
    assert body["evaluation"]["per_class_iou"]["trees"] == 0.5
