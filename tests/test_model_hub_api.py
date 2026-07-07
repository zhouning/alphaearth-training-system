import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def _write_api_test_geotiff(path: Path, *, bands: int = 18) -> Path:
    data = np.zeros((bands, 8, 8), dtype=np.float32)
    for band in range(bands):
        data[band] = (band + 1) / max(bands, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(100.0, 40.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)
    return path


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


def test_model_hub_returns_prithvi_crop_model_details():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/model-hub/models/prithvi_crop_classification_arcgis_style")

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "prithvi_crop_classification_arcgis_style"
    assert body["task_type"] == "crop_classification"
    assert body["status"] == "demo_only"
    assert body["package_profile"]["package_type"] == "arcgis_style_pretrained_imagery_model"
    assert body["package_profile"]["input_profile"]["raster_profile"] == "18_band_hls_multitemporal_composite"
    assert body["package_profile"]["output_profile"]["primary_output"] == "categorical crop raster"


def test_model_hub_runs_prithvi_crop_cached_demo_job(monkeypatch, tmp_path: Path):
    from app.main import app
    import app.services.model_hub_crop as crop_service

    crop_dir = tmp_path / "prithvi_crop_demo"
    crop_dir.mkdir()
    (crop_dir / "crop_preview.png").write_bytes(b"png")
    (crop_dir / "crop_polygons.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    (crop_dir / "crop_summary.csv").write_text(
        "class,pixels,fraction\ncorn,6400,0.64\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(crop_service, "default_crop_demo_dir", lambda: crop_dir)

    client = TestClient(app)
    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "cached_demo",
            "options": {"output_formats": ["png", "geojson", "csv"]},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["task"] == "crop_classification"
    assert body["result"]["summary"]["dominant_class"] == "corn"
    assert {artifact["kind"] for artifact in body["artifacts"]} == {"png", "geojson", "csv"}


def test_model_hub_runs_prithvi_crop_upload_raster_demo_job(tmp_path: Path):
    from app.main import app

    raster_path = _write_api_test_geotiff(tmp_path / "crop_18band.tif", bands=18)
    output_dir = tmp_path / "outputs"
    client = TestClient(app)

    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "upload_raster_demo",
            "options": {
                "raster_path": str(raster_path),
                "output_dir": str(output_dir),
                "tile_size": 4,
                "stride": 4,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["input_mode"] == "upload_raster_demo"
    assert body["result"]["validation"]["band_count"] == 18
    assert {"geotiff", "csv", "geojson", "manifest"}.issubset(
        {artifact["kind"] for artifact in body["artifacts"]}
    )
    assert any("validated 18-band Prithvi crop raster" in log for log in body["logs"])
    assert any("deterministic tiled crop classification" in log for log in body["logs"])


def test_model_hub_fails_prithvi_crop_upload_raster_demo_for_wrong_band_count(tmp_path: Path):
    from app.main import app

    raster_path = _write_api_test_geotiff(tmp_path / "crop_6band.tif", bands=6)
    client = TestClient(app)

    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "upload_raster_demo",
            "options": {"raster_path": str(raster_path)},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "requires 18 bands" in body["error"]


def test_model_hub_fails_prithvi_crop_upload_raster_demo_for_unsafe_raster_path(tmp_path: Path):
    from app.main import app

    unsafe_dir = repo_root / "ae_backend" / "not_allowed_api_inputs"
    unsafe_dir.mkdir(exist_ok=True)
    try:
        raster_path = _write_api_test_geotiff(unsafe_dir / "crop_18band.tif", bands=18)
        client = TestClient(app)

        response = client.post(
            "/api/ae/model-hub/jobs",
            json={
                "model_id": "prithvi_crop_classification_arcgis_style",
                "input_mode": "upload_raster_demo",
                "options": {
                    "raster_path": str(raster_path),
                    "output_dir": str(tmp_path / "outputs"),
                },
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed"
        assert "raster_path" in body["error"]
        assert "allowed" in body["error"]
    finally:
        shutil.rmtree(unsafe_dir, ignore_errors=True)


def test_model_hub_fails_prithvi_crop_upload_raster_demo_for_unsafe_output_dir(tmp_path: Path):
    from app.main import app

    raster_path = _write_api_test_geotiff(tmp_path / "crop_18band.tif", bands=18)
    unsafe_output_dir = repo_root / "ae_backend" / "not_allowed_api_outputs"
    client = TestClient(app)

    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "upload_raster_demo",
            "options": {
                "raster_path": str(raster_path),
                "output_dir": str(unsafe_output_dir),
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "output_dir" in body["error"]
    assert "allowed" in body["error"]
def test_model_hub_api_preserves_all_runtime_logs(monkeypatch):
    from app.main import app
    import app.api.model_hub as model_hub_api

    def fake_run_model_hub_job(*, model_id, input_mode, options):
        assert model_id == "prithvi_crop_classification_arcgis_style"
        assert input_mode == "upload_raster_demo"
        return {
            "result": {"task": "crop_classification", "summary": {"dominant_class": "corn"}},
            "artifacts": [],
            "logs": ["first runtime log", "second runtime log"],
        }

    monkeypatch.setattr(model_hub_api, "run_model_hub_job", fake_run_model_hub_job)
    client = TestClient(app)

    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "upload_raster_demo",
            "options": {"raster_path": "unused.tif"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["logs"][-2:] == ["first runtime log", "second runtime log"]


def test_model_hub_returns_paper12_summary():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/model-hub/paper12-summary")

    assert response.status_code == 200
    body = response.json()
    assert body["paper"] == "paper12"
    assert body["readiness_counts"]["ready"] >= 1
    assert body["readiness_counts"]["demo_only"] >= 1
    assert body["readiness_counts"]["planned"] >= 1

    benchmarks = {item["id"]: item for item in body["benchmarks"]}
    assert benchmarks["eurosat_channel_bridge"]["best_method"] == "learned_bridge_houlsby"
    assert benchmarks["eurosat_channel_bridge"]["metric"] == "overall_accuracy"
    assert benchmarks["eurosat_channel_bridge"]["best_value"] > 0.9
    assert benchmarks["landcoverai_segmentation"]["best_method"] == "houlsby"
    assert benchmarks["landcoverai_segmentation"]["metric"] == "mIoU"
    assert benchmarks["landcoverai_segmentation"]["best_value"] > 0.64

    crop = {
        item["model_id"]: item
        for item in body["capabilities"]
    }["prithvi_crop_classification_arcgis_style"]
    assert crop["readiness"] == "demo_only"
    assert crop["arcgis_replacement_status"] == "not_yet"
    assert "No validated crop checkpoint" in crop["reason"]


def test_model_hub_paper12_summary_reports_missing_optional_results(
    monkeypatch,
    tmp_path: Path,
):
    from app.main import app
    import app.services.paper12_summary as paper12_summary

    monkeypatch.setattr(paper12_summary, "PAPER12_RESULTS_DIR", tmp_path)

    client = TestClient(app)
    response = client.get("/api/ae/model-hub/paper12-summary")

    assert response.status_code == 200
    body = response.json()
    missing = [item for item in body["benchmarks"] if item.get("status") == "missing"]
    assert missing
    assert any("missing" in item["note"].lower() for item in missing)

def test_system_capabilities_endpoint_reports_operational_readiness():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/system/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "AlphaEarth System"
    assert set(body) >= {
        "generated_at",
        "readiness_counts",
        "summary",
        "capabilities",
        "evidence_sources",
    }
    assert body["readiness_counts"]["ready"] >= 1
    assert body["readiness_counts"]["demo_only"] >= 1
    assert body["readiness_counts"]["planned"] >= 1
    assert body["summary"]["runnable_models"] >= 1
    assert body["summary"]["demo_workflows"] >= 1
    assert body["summary"]["arcgis_replacement_ready"] is False

    capabilities = {item["id"]: item for item in body["capabilities"]}
    lulc = capabilities["lulc_6class_prithvi_houlsby"]
    assert lulc["readiness"] == "ready"
    assert lulc["workflow_level"] == "runnable_and_evaluable"
    assert lulc["checkpoint"]["configured"] is True
    assert "demo_patch" in lulc["runtime_modes"]
    assert any(item["label"] == "mIoU" for item in lulc["evidence"])

    crop = capabilities["prithvi_crop_classification_arcgis_style"]
    assert crop["readiness"] == "demo_only"
    assert crop["workflow_level"] == "contract_demo"
    assert crop["checkpoint"]["configured"] is False
    assert crop["arcgis_replacement"]["status"] == "not_ready"
    assert "No validated crop checkpoint" in crop["arcgis_replacement"]["reason"]
    assert "upload_raster_demo" in crop["runtime_modes"]


def test_system_capabilities_tolerates_missing_optional_evidence(monkeypatch, tmp_path: Path):
    from app.main import app
    import app.services.system_capabilities as system_capabilities

    monkeypatch.setattr(system_capabilities, "PAPER12_RESULTS_DIR", tmp_path)

    client = TestClient(app)
    response = client.get("/api/ae/system/capabilities")

    assert response.status_code == 200
    body = response.json()
    missing_sources = [
        item
        for item in body["evidence_sources"]
        if item["kind"] == "paper12_benchmark" and item["available"] is False
    ]
    assert missing_sources
    assert all("missing" in item["note"].lower() for item in missing_sources)


def test_system_verification_endpoint_reports_contract_checks():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/system/verification")

    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "AlphaEarth System"
    assert body["overall_status"] in {"pass", "warning", "fail"}
    assert set(body["summary"]) == {"pass", "warning", "fail", "not_applicable"}
    assert set(body) >= {"generated_at", "capabilities", "checks", "notes"}
    assert body["checks"]
    assert body["notes"]

    capabilities = {item["id"]: item for item in body["capabilities"]}
    lulc = capabilities["lulc_6class_prithvi_houlsby"]
    assert lulc["overall_status"] in {"pass", "warning"}
    assert lulc["blocking_issues"] == []
    assert any(check_id.endswith(":checkpoint_configured") for check_id in lulc["checks"])

    crop_id = "prithvi_crop_classification_arcgis_style"
    crop = capabilities[crop_id]
    assert crop["overall_status"] in {"pass", "warning"}
    assert crop["blocking_issues"] == []
    assert any(check_id.endswith(":arcgis_replacement_guard") for check_id in crop["checks"])

    crop_checks = {
        check["id"]: check
        for check in body["checks"]
        if check["capability_id"] == crop_id
    }
    guard = crop_checks[f"{crop_id}:arcgis_replacement_guard"]
    assert guard["category"] == "replacement_boundary"
    assert guard["status"] == "pass"
    assert "not a validated ArcGIS replacement" in guard["detail"]
    assert "validated Prithvi crop checkpoint" in " ".join(crop["next_actions"])


def test_system_verification_does_not_fail_planned_models_for_missing_checkpoints():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/system/verification")

    assert response.status_code == 200
    body = response.json()
    capabilities = {item["id"]: item for item in body["capabilities"]}
    planned = capabilities["building_extraction_prithvi"]
    assert planned["overall_status"] in {"not_applicable", "warning"}
    assert all("checkpoint" not in issue.lower() for issue in planned["blocking_issues"])

    planned_checks = [
        check
        for check in body["checks"]
        if check["capability_id"] == "building_extraction_prithvi"
    ]
    checkpoint_checks = [
        check for check in planned_checks if check["category"] == "checkpoint_configuration"
    ]
    assert checkpoint_checks
    assert all(check["status"] == "not_applicable" for check in checkpoint_checks)


def test_system_verification_reports_missing_optional_evidence_as_warnings(
    monkeypatch,
    tmp_path: Path,
):
    from app.main import app
    import app.services.system_capabilities as system_capabilities

    monkeypatch.setattr(system_capabilities, "PAPER12_RESULTS_DIR", tmp_path)

    client = TestClient(app)
    response = client.get("/api/ae/system/verification")

    assert response.status_code == 200
    body = response.json()
    evidence_checks = [
        check
        for check in body["checks"]
        if check["category"] == "evidence_source"
    ]
    assert evidence_checks
    assert any(check["status"] == "warning" for check in evidence_checks)
    assert all(check["status"] != "fail" for check in evidence_checks)
    assert any("Missing optional evidence file" in check["detail"] for check in evidence_checks)


def test_system_evidence_endpoint_reports_drilldown_payload():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/ae/system/evidence")

    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "AlphaEarth System"
    assert set(body) >= {"generated_at", "summary", "checks", "artifacts", "notes"}
    assert set(body["summary"]) == {"available", "missing", "previewable", "blocked"}
    assert body["checks"]
    assert body["artifacts"]
    assert body["notes"]

    evidence_items = [
        artifact
        for check in body["checks"]
        for artifact in check["evidence"]
    ]
    assert evidence_items
    assert any(item["status"] == "available" for item in evidence_items)
    assert any(item["ref"] == "model_hub_registry" and item["kind"] == "registry" for item in evidence_items)
    assert all(not Path(item.get("safe_path", "relative")).is_absolute() for item in evidence_items)


def test_system_evidence_reports_missing_optional_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    from app.main import app
    import app.services.system_capabilities as system_capabilities

    monkeypatch.setattr(system_capabilities, "PAPER12_RESULTS_DIR", tmp_path)

    client = TestClient(app)
    response = client.get("/api/ae/system/evidence")

    assert response.status_code == 200
    body = response.json()
    missing = [
        artifact
        for artifact in body["artifacts"]
        if artifact["status"] == "missing"
    ]
    assert missing
    assert body["summary"]["missing"] >= len(missing)
    assert all(item["safe_path"].startswith("paper12_results/") for item in missing)
    assert all("optional" in item["message"].lower() or "referenced" in item["message"].lower() for item in missing)


def test_system_evidence_blocks_unsafe_refs_without_leaking_absolute_paths(
    monkeypatch,
):
    import json

    from app.main import app
    import app.services.system_evidence as system_evidence

    absolute_ref = str(repo_root / "pyproject.toml")

    def fake_build_system_verification(registry):
        return {
            "system": "AlphaEarth System",
            "generated_at": "2026-07-07T00:00:00+00:00",
            "checks": [
                {
                    "id": "system:unsafe_ref",
                    "capability_id": "system",
                    "status": "warning",
                    "title": "Unsafe evidence ref",
                    "evidence_refs": ["../outside.txt", absolute_ref],
                    "remediation": "Remove unsafe evidence refs.",
                }
            ],
        }

    monkeypatch.setattr(
        system_evidence.system_verification,
        "build_system_verification",
        fake_build_system_verification,
    )

    client = TestClient(app)
    response = client.get("/api/ae/system/evidence")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["blocked"] == 2
    blocked = [artifact for artifact in body["artifacts"] if artifact["status"] == "blocked"]
    assert len(blocked) == 2
    assert all("safe_path" not in item for item in blocked)
    payload = json.dumps(body)
    assert str(repo_root) not in payload
    assert absolute_ref not in payload


def test_system_evidence_previews_json_and_csv_from_allowed_roots(
    monkeypatch,
    tmp_path: Path,
):
    from app.main import app
    import app.services.system_evidence as system_evidence

    evidence_dir = tmp_path / "paper12_results"
    evidence_dir.mkdir()
    (evidence_dir / "preview.json").write_text(
        '{"metrics":{"overall_accuracy":0.91},"items":[1,2,3]}',
        encoding="utf-8",
    )
    (evidence_dir / "table.csv").write_text(
        "class,pixels,fraction\ncorn,64,0.64\nsoybean,36,0.36\n",
        encoding="utf-8",
    )

    def fake_build_system_verification(registry):
        return {
            "system": "AlphaEarth System",
            "generated_at": "2026-07-07T00:00:00+00:00",
            "checks": [
                {
                    "id": "system:preview_refs",
                    "capability_id": "system",
                    "status": "pass",
                    "title": "Preview refs",
                    "evidence_refs": [
                        "paper12_results/preview.json",
                        "paper12_results/table.csv",
                    ],
                    "remediation": None,
                }
            ],
        }

    monkeypatch.setattr(system_evidence, "PROJECT_ROOT_PATH", tmp_path.resolve())
    monkeypatch.setattr(
        system_evidence.system_verification,
        "build_system_verification",
        fake_build_system_verification,
    )

    client = TestClient(app)
    response = client.get("/api/ae/system/evidence")

    assert response.status_code == 200
    body = response.json()
    by_ref = {artifact["ref"]: artifact for artifact in body["artifacts"]}
    json_artifact = by_ref["paper12_results/preview.json"]
    csv_artifact = by_ref["paper12_results/table.csv"]

    assert json_artifact["previewable"] is True
    assert json_artifact["preview"]["type"] == "json"
    assert json_artifact["preview"]["content"]["metrics"]["overall_accuracy"] == 0.91
    assert csv_artifact["previewable"] is True
    assert csv_artifact["preview"]["type"] == "csv"
    assert csv_artifact["preview"]["header"] == ["class", "pixels", "fraction"]
    assert csv_artifact["preview"]["rows"][0] == ["corn", "64", "0.64"]
