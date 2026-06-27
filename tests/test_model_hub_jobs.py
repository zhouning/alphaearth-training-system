import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_job_store_creates_and_retrieves_pending_job():
    from app.services.model_hub_jobs import ModelHubJobStore

    store = ModelHubJobStore()
    job = store.create_job(
        model_id="lulc_6class_prithvi_houlsby",
        input_mode="demo_patch",
        options={"output_formats": ["png", "csv"]},
    )

    loaded = store.get_job(job["job_id"])
    assert loaded["status"] == "pending"
    assert loaded["model_id"] == "lulc_6class_prithvi_houlsby"
    assert loaded["input_mode"] == "demo_patch"


def test_job_store_marks_success_with_artifacts():
    from app.services.model_hub_jobs import ModelHubJobStore

    store = ModelHubJobStore()
    job = store.create_job("semantic_change_prithvi", "cached_demo", {})
    store.mark_running(job["job_id"], log="started")
    store.mark_succeeded(
        job["job_id"],
        result={"summary": {"changed_pairs": 10}},
        artifacts=[{"kind": "geojson", "path": "results/model_hub/change.geojson"}],
        log="finished",
    )

    loaded = store.get_job(job["job_id"])
    assert loaded["status"] == "succeeded"
    assert loaded["result"]["summary"]["changed_pairs"] == 10
    assert loaded["artifacts"][0]["kind"] == "geojson"
    assert loaded["logs"] == ["started", "finished"]


def test_job_store_marks_failure():
    from app.services.model_hub_jobs import ModelHubJobStore

    store = ModelHubJobStore()
    job = store.create_job("water_flood_prithvi", "upload", {})
    store.mark_failed(job["job_id"], error="checkpoint missing")

    loaded = store.get_job(job["job_id"])
    assert loaded["status"] == "failed"
    assert loaded["error"] == "checkpoint missing"


def test_job_store_returns_independent_job_snapshots():
    from app.services.model_hub_jobs import ModelHubJobStore

    store = ModelHubJobStore()
    options = {"output_formats": ["png"]}
    job = store.create_job("semantic_change_prithvi", "cached_demo", options)

    options["output_formats"].append("csv")
    job["logs"].append("client mutation")
    loaded = store.get_job(job["job_id"])

    assert loaded["options"] == {"output_formats": ["png"]}
    assert loaded["logs"] == []

    artifacts = [{"kind": "geojson", "path": "results/model_hub/change.geojson"}]
    result = {"summary": {"changed_pairs": 10}}
    store.mark_succeeded(job["job_id"], result=result, artifacts=artifacts)

    artifacts[0]["path"] = "changed.geojson"
    result["summary"]["changed_pairs"] = 99
    succeeded = store.get_job(job["job_id"])

    assert succeeded["artifacts"][0]["path"] == "results/model_hub/change.geojson"
    assert succeeded["result"]["summary"]["changed_pairs"] == 10


def test_job_store_marks_success_with_multiple_runtime_logs():
    from app.services.model_hub_jobs import ModelHubJobStore

    store = ModelHubJobStore()
    job = store.create_job("prithvi_crop_classification_arcgis_style", "upload_raster_demo", {})
    store.mark_running(job["job_id"], log="job accepted")
    store.mark_succeeded(
        job["job_id"],
        result={"summary": {"dominant_class": "corn"}},
        artifacts=[],
        logs=[
            "validated 18-band Prithvi crop raster",
            "ran deterministic tiled crop classification contract demo",
        ],
    )

    loaded = store.get_job(job["job_id"])
    assert loaded["status"] == "succeeded"
    assert loaded["logs"] == [
        "job accepted",
        "validated 18-band Prithvi crop raster",
        "ran deterministic tiled crop classification contract demo",
    ]
