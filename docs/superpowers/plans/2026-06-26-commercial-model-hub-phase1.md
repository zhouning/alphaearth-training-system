# Commercial Model Hub Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 Model Hub foundation so natural-resource users can list commercial model packages, start LULC/change jobs, and inspect product-ready artifacts through shared backend and frontend entry points.

**Architecture:** Add a registry-driven model catalog, a lightweight in-process job store, a shared model-hub API namespace, and runtime wrappers for existing LULC and Linhe change assets. Keep the first implementation local and deterministic; persist outputs under `results/model_hub/` and shape all interfaces so Celery/PostGIS/OBS can replace the in-memory/local pieces later.

**Tech Stack:** FastAPI, Pydantic/dataclasses as already used in backend style, Python standard library JSON/UUID/pathlib, NumPy/Pillow for patch tests, rasterio for raster-pipeline tests, Vue single-file HTML frontend, pytest + FastAPI TestClient.

---

## Source Spec

- Spec: `docs/superpowers/specs/2026-06-26-commercial-model-hub-phase1-design.md`
- Current branch: `paper12-results-colab-20260619`
- First runnable capabilities: LULC segmentation and change detection.
- First planned catalog entries: LULC, building extraction, road/hard-surface, water/flood, semantic change.

## File Structure

Create these backend files:

- `ae_backend/app/services/model_hub_registry.py`
  Loads and validates model metadata from JSON. Responsible only for registry parsing, model lookup, status/readiness calculation, and serializable dictionaries.

- `ae_backend/app/services/model_hub_jobs.py`
  Owns local job lifecycle: create job, mark running/succeeded/failed, attach artifacts, expose logs. It does not run models by itself.

- `ae_backend/app/services/model_hub_runtime.py`
  Dispatches a model-hub job to a task runtime. First supports LULC patch inference via the existing `LULCInferenceService` and demo change-artifact jobs via `model_hub_change.py`.

- `ae_backend/app/services/model_hub_change.py`
  Wraps existing Linhe change outputs into a model-hub artifact manifest and summary payload.

- `ae_backend/app/services/raster_pipeline.py`
  Shared raster utility functions: tile grid generation, class-area statistics, array stitching, and GeoTIFF export.

- `ae_backend/app/api/model_hub.py`
  FastAPI routes under `/api/ae/model-hub`.

- `ae_backend/app/data/model_hub_models.json`
  Phase 1 model metadata registry.

Modify these existing files:

- `ae_backend/app/main.py`
  Include the new model-hub router.

- `ae_frontend/index.html`
  Add a Model Hub navigation entry and a simple model-center page wired to `/api/ae/model-hub/models`.

Create these tests:

- `tests/test_model_hub_registry.py`
- `tests/test_model_hub_api.py`
- `tests/test_model_hub_jobs.py`
- `tests/test_raster_pipeline.py`
- `tests/test_model_hub_change.py`
- `tests/test_model_hub_frontend_entry.py`

## Task 1: Model Registry

**Files:**
- Create: `ae_backend/app/services/model_hub_registry.py`
- Create: `ae_backend/app/data/model_hub_models.json`
- Test: `tests/test_model_hub_registry.py`

- [ ] **Step 1: Write registry tests**

Create `tests/test_model_hub_registry.py`:

```python
import json
import sys
from pathlib import Path

import pytest


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def _write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "model_id": "lulc_6class_prithvi_houlsby",
                    "display_name": "LULC 6-class Prithvi Houlsby",
                    "task_type": "semantic_segmentation",
                    "backbone": "Prithvi-100M",
                    "adapter": "houlsby",
                    "checkpoint_path": "linhe_lulc/houlsby__rgb_3band__seed123.pt",
                    "input_spec": {"bands": ["red", "green", "blue"], "tile_size": 128},
                    "output_spec": {"formats": ["png", "geotiff", "geojson", "csv"]},
                    "class_schema": ["background", "built", "crops", "trees", "water", "rangeland_bare"],
                    "metrics": {"mIoU": 0.2971},
                    "trained_region": "Linhe County demo corpus",
                    "supported_sensors": ["RGB"],
                    "license": "internal-demo",
                    "status": "ready",
                    "example_inputs": ["linhe_rgb_patch"],
                },
                {
                    "model_id": "semantic_change_prithvi",
                    "display_name": "Semantic Change Detection",
                    "task_type": "change_detection",
                    "backbone": "Prithvi-100M plus visual anomaly scoring",
                    "adapter": "mixed",
                    "checkpoint_path": None,
                    "input_spec": {"date_pair": True, "bands": ["red", "green", "blue"]},
                    "output_spec": {"formats": ["png", "geojson", "csv"]},
                    "class_schema": ["unchanged", "changed"],
                    "metrics": {"top_pca_score": 0.386},
                    "trained_region": "Linhe County demo corpus",
                    "supported_sensors": ["RGB"],
                    "license": "internal-demo",
                    "status": "demo_only",
                    "example_inputs": ["linhe_2025Q1_2025Q4"],
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def test_load_model_registry_validates_and_indexes_entries(tmp_path: Path):
    from app.services.model_hub_registry import load_model_registry

    registry_path = tmp_path / "models.json"
    _write_registry(registry_path)

    registry = load_model_registry(registry_path)

    assert len(registry.models) == 2
    assert registry.get_model("lulc_6class_prithvi_houlsby").task_type == "semantic_segmentation"
    assert registry.get_model("semantic_change_prithvi").status == "demo_only"
    assert registry.to_public_dict()["models"][0]["model_id"] == "lulc_6class_prithvi_houlsby"


def test_load_model_registry_rejects_duplicate_model_ids(tmp_path: Path):
    from app.services.model_hub_registry import RegistryValidationError, load_model_registry

    registry_path = tmp_path / "models.json"
    _write_registry(registry_path)
    records = json.loads(registry_path.read_text(encoding="utf-8"))
    records.append(records[0])
    registry_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(RegistryValidationError, match="Duplicate model_id"):
        load_model_registry(registry_path)


def test_load_model_registry_rejects_missing_required_field(tmp_path: Path):
    from app.services.model_hub_registry import RegistryValidationError, load_model_registry

    registry_path = tmp_path / "models.json"
    _write_registry(registry_path)
    records = json.loads(registry_path.read_text(encoding="utf-8"))
    del records[0]["task_type"]
    registry_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(RegistryValidationError, match="task_type"):
        load_model_registry(registry_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_model_hub_registry.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'app.services.model_hub_registry'`.

- [ ] **Step 3: Implement registry service**

Create `ae_backend/app/services/model_hub_registry.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "model_id",
    "display_name",
    "task_type",
    "backbone",
    "adapter",
    "input_spec",
    "output_spec",
    "class_schema",
    "metrics",
    "trained_region",
    "supported_sensors",
    "license",
    "status",
    "example_inputs",
}

VALID_STATUSES = {"ready", "demo_only", "planned", "not_configured"}


class RegistryValidationError(ValueError):
    """Raised when model-hub registry metadata is invalid."""


@dataclass(frozen=True)
class ModelHubEntry:
    model_id: str
    display_name: str
    task_type: str
    backbone: str
    adapter: str
    checkpoint_path: str | None
    input_spec: dict[str, Any]
    output_spec: dict[str, Any]
    class_schema: list[str]
    metrics: dict[str, Any]
    trained_region: str
    supported_sensors: list[str]
    license: str
    status: str
    example_inputs: list[str]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ModelHubEntry":
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            raise RegistryValidationError(f"Missing required fields for model registry entry: {', '.join(missing)}")
        status = str(record["status"])
        if status not in VALID_STATUSES:
            raise RegistryValidationError(f"Invalid status for {record.get('model_id')}: {status}")
        return cls(
            model_id=str(record["model_id"]),
            display_name=str(record["display_name"]),
            task_type=str(record["task_type"]),
            backbone=str(record["backbone"]),
            adapter=str(record["adapter"]),
            checkpoint_path=record.get("checkpoint_path"),
            input_spec=dict(record["input_spec"]),
            output_spec=dict(record["output_spec"]),
            class_schema=list(record["class_schema"]),
            metrics=dict(record["metrics"]),
            trained_region=str(record["trained_region"]),
            supported_sensors=list(record["supported_sensors"]),
            license=str(record["license"]),
            status=status,
            example_inputs=list(record["example_inputs"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "backbone": self.backbone,
            "adapter": self.adapter,
            "checkpoint_path": self.checkpoint_path,
            "input_spec": self.input_spec,
            "output_spec": self.output_spec,
            "class_schema": self.class_schema,
            "metrics": self.metrics,
            "trained_region": self.trained_region,
            "supported_sensors": self.supported_sensors,
            "license": self.license,
            "status": self.status,
            "example_inputs": self.example_inputs,
        }


class ModelHubRegistry:
    def __init__(self, models: list[ModelHubEntry]):
        self.models = models
        self._by_id = {entry.model_id: entry for entry in models}

    def get_model(self, model_id: str) -> ModelHubEntry:
        if model_id not in self._by_id:
            raise KeyError(model_id)
        return self._by_id[model_id]

    def to_public_dict(self) -> dict[str, Any]:
        return {"models": [entry.to_dict() for entry in self.models]}


def load_model_registry(path: str | Path) -> ModelHubRegistry:
    registry_path = Path(path)
    records = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RegistryValidationError("Model registry must be a JSON list.")
    entries = [ModelHubEntry.from_record(record) for record in records]
    seen: set[str] = set()
    for entry in entries:
        if entry.model_id in seen:
            raise RegistryValidationError(f"Duplicate model_id: {entry.model_id}")
        seen.add(entry.model_id)
    return ModelHubRegistry(entries)
```

- [ ] **Step 4: Add default model registry JSON**

Create `ae_backend/app/data/model_hub_models.json`:

```json
[
  {
    "model_id": "lulc_6class_prithvi_houlsby",
    "display_name": "LULC 6-class Prithvi Houlsby",
    "task_type": "semantic_segmentation",
    "backbone": "Prithvi-100M",
    "adapter": "houlsby",
    "checkpoint_path": "linhe_lulc/houlsby__rgb_3band__seed123.pt",
    "input_spec": {"bands": ["red", "green", "blue"], "tile_size": 128, "input_mode": "rgb_patch_or_raster"},
    "output_spec": {"formats": ["png", "geotiff", "geojson", "csv"]},
    "class_schema": ["background", "built", "crops", "trees", "water", "rangeland_bare"],
    "metrics": {"mIoU": 0.2971, "source": "linhe_results/linhe_lulc_seg.json"},
    "trained_region": "Linhe County demo corpus",
    "supported_sensors": ["RGB"],
    "license": "internal-demo",
    "status": "ready",
    "example_inputs": ["linhe_rgb_patch"]
  },
  {
    "model_id": "building_extraction_prithvi",
    "display_name": "Building Extraction Prithvi",
    "task_type": "binary_segmentation",
    "backbone": "Prithvi-100M",
    "adapter": "houlsby",
    "checkpoint_path": null,
    "input_spec": {"bands": ["red", "green", "blue"], "tile_size": 128, "input_mode": "rgb_patch_or_raster"},
    "output_spec": {"formats": ["png", "geotiff", "geojson", "csv"]},
    "class_schema": ["background", "building"],
    "metrics": {"label_source": "OSM weak supervision"},
    "trained_region": "Linhe County weak-label corpus",
    "supported_sensors": ["RGB"],
    "license": "internal-demo",
    "status": "planned",
    "example_inputs": ["linhe_rgb_patch"]
  },
  {
    "model_id": "road_hardscape_prithvi",
    "display_name": "Road and Hard Surface Prithvi",
    "task_type": "semantic_segmentation",
    "backbone": "Prithvi-100M",
    "adapter": "houlsby",
    "checkpoint_path": null,
    "input_spec": {"bands": ["red", "green", "blue"], "tile_size": 128, "input_mode": "rgb_patch_or_raster"},
    "output_spec": {"formats": ["png", "geotiff", "geojson", "csv"]},
    "class_schema": ["background", "road_hardscape"],
    "metrics": {"label_source": "planned vector rasterization"},
    "trained_region": "Linhe County planned corpus",
    "supported_sensors": ["RGB"],
    "license": "internal-demo",
    "status": "planned",
    "example_inputs": ["linhe_rgb_patch"]
  },
  {
    "model_id": "water_flood_prithvi",
    "display_name": "Water and Flood Prithvi",
    "task_type": "semantic_segmentation",
    "backbone": "Prithvi-100M",
    "adapter": "houlsby",
    "checkpoint_path": null,
    "input_spec": {"bands": ["red", "green", "blue"], "tile_size": 128, "input_mode": "rgb_or_multispectral"},
    "output_spec": {"formats": ["png", "geotiff", "geojson", "csv"]},
    "class_schema": ["background", "water", "flood_candidate"],
    "metrics": {"label_source": "planned water/flood labels"},
    "trained_region": "planned",
    "supported_sensors": ["RGB", "Sentinel-2", "SAR"],
    "license": "internal-demo",
    "status": "planned",
    "example_inputs": ["linhe_rgb_patch"]
  },
  {
    "model_id": "semantic_change_prithvi",
    "display_name": "Semantic and Visual Change Detection",
    "task_type": "change_detection",
    "backbone": "Prithvi-100M plus visual anomaly scoring",
    "adapter": "mixed",
    "checkpoint_path": null,
    "input_spec": {"date_pair": true, "bands": ["red", "green", "blue"], "input_mode": "paired_rgb_rasters_or_cached_linhe_pairs"},
    "output_spec": {"formats": ["png", "geojson", "csv"]},
    "class_schema": ["unchanged", "changed"],
    "metrics": {"top_pca_score": 0.386, "source": "results/linhe_change"},
    "trained_region": "Linhe County demo corpus",
    "supported_sensors": ["RGB"],
    "license": "internal-demo",
    "status": "demo_only",
    "example_inputs": ["linhe_2025Q1_2025Q4"]
  }
]
```

- [ ] **Step 5: Run registry tests**

Run:

```bash
python -m pytest tests/test_model_hub_registry.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add ae_backend/app/services/model_hub_registry.py ae_backend/app/data/model_hub_models.json tests/test_model_hub_registry.py
git commit -m "feat: add model hub registry"
```

## Task 2: Model Catalog API

**Files:**
- Create: `ae_backend/app/api/model_hub.py`
- Modify: `ae_backend/app/main.py`
- Test: `tests/test_model_hub_api.py`

- [ ] **Step 1: Write API tests for model listing and details**

Create `tests/test_model_hub_api.py`:

```python
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
```

- [ ] **Step 2: Run API tests to verify they fail**

Run:

```bash
python -m pytest tests/test_model_hub_api.py -q
```

Expected: fail with route returning `404 Not Found`.

- [ ] **Step 3: Implement model-hub API routes**

Create `ae_backend/app/api/model_hub.py`:

```python
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import PROJECT_ROOT
from app.services.model_hub_registry import ModelHubRegistry, load_model_registry


router = APIRouter()
DEFAULT_REGISTRY_PATH = Path(PROJECT_ROOT) / "ae_backend" / "app" / "data" / "model_hub_models.json"


@lru_cache(maxsize=1)
def get_model_registry() -> ModelHubRegistry:
    return load_model_registry(DEFAULT_REGISTRY_PATH)


@router.get("/models")
def list_models():
    return get_model_registry().to_public_dict()


@router.get("/models/{model_id}")
def get_model(model_id: str):
    try:
        return get_model_registry().get_model(model_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}") from exc
```

- [ ] **Step 4: Register router in app**

Modify `ae_backend/app/main.py`:

```python
from app.api import pipeline, training, satellites, areas, models, results, inference, model_hub
```

Add this block after the inference router:

```python
app.include_router(
    model_hub.router,
    prefix=f"{settings.API_V1_STR}/model-hub",
    tags=["model-hub"]
)
```

- [ ] **Step 5: Run API tests**

Run:

```bash
python -m pytest tests/test_model_hub_api.py tests/test_model_hub_registry.py -q
```

Expected: `6 passed`.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add ae_backend/app/api/model_hub.py ae_backend/app/main.py tests/test_model_hub_api.py
git commit -m "feat: expose model hub catalog api"
```

## Task 3: Local Job Store And Job API

**Files:**
- Create: `ae_backend/app/services/model_hub_jobs.py`
- Modify: `ae_backend/app/api/model_hub.py`
- Test: `tests/test_model_hub_jobs.py`
- Test: `tests/test_model_hub_api.py`

- [ ] **Step 1: Write job-store unit tests**

Create `tests/test_model_hub_jobs.py`:

```python
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
```

- [ ] **Step 2: Extend API tests for job creation and lookup**

Append to `tests/test_model_hub_api.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify failures**

Run:

```bash
python -m pytest tests/test_model_hub_jobs.py tests/test_model_hub_api.py -q
```

Expected: fail with missing `model_hub_jobs` and missing job routes.

- [ ] **Step 4: Implement local job store**

Create `ae_backend/app/services/model_hub_jobs.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelHubJobStore:
    def __init__(self):
        self._jobs: dict[str, dict] = {}

    def create_job(self, model_id: str, input_mode: str, options: dict) -> dict:
        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "model_id": model_id,
            "input_mode": input_mode,
            "options": options,
            "status": "pending",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "result": None,
            "artifacts": [],
            "logs": [],
            "error": None,
        }
        self._jobs[job_id] = job
        return dict(job)

    def get_job(self, job_id: str) -> dict:
        if job_id not in self._jobs:
            raise KeyError(job_id)
        return dict(self._jobs[job_id])

    def mark_running(self, job_id: str, log: str | None = None) -> None:
        job = self._jobs[job_id]
        job["status"] = "running"
        job["updated_at"] = _utc_now()
        if log:
            job["logs"].append(log)

    def mark_succeeded(self, job_id: str, result: dict, artifacts: list[dict], log: str | None = None) -> None:
        job = self._jobs[job_id]
        job["status"] = "succeeded"
        job["result"] = result
        job["artifacts"] = artifacts
        job["updated_at"] = _utc_now()
        if log:
            job["logs"].append(log)

    def mark_failed(self, job_id: str, error: str) -> None:
        job = self._jobs[job_id]
        job["status"] = "failed"
        job["error"] = error
        job["updated_at"] = _utc_now()
        job["logs"].append(error)
```

- [ ] **Step 5: Add job routes**

Modify `ae_backend/app/api/model_hub.py`:

```python
from pydantic import BaseModel, Field

from app.services.model_hub_jobs import ModelHubJobStore


class ModelHubJobRequest(BaseModel):
    model_id: str
    input_mode: str = Field(default="cached_demo")
    options: dict = Field(default_factory=dict)


JOB_STORE = ModelHubJobStore()
```

Add these routes:

```python
@router.post("/jobs")
def create_job(request: ModelHubJobRequest):
    try:
        get_model_registry().get_model(request.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {request.model_id}") from exc
    return JOB_STORE.create_job(
        model_id=request.model_id,
        input_mode=request.input_mode,
        options=request.options,
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return JOB_STORE.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}") from exc
```

- [ ] **Step 6: Run job tests**

Run:

```bash
python -m pytest tests/test_model_hub_jobs.py tests/test_model_hub_api.py -q
```

Expected: `8 passed`.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add ae_backend/app/services/model_hub_jobs.py ae_backend/app/api/model_hub.py tests/test_model_hub_jobs.py tests/test_model_hub_api.py
git commit -m "feat: add model hub local jobs"
```

## Task 4: Runtime Dispatcher And LULC Patch Jobs

**Files:**
- Create: `ae_backend/app/services/model_hub_runtime.py`
- Modify: `ae_backend/app/api/model_hub.py`
- Test: `tests/test_model_hub_api.py`

- [ ] **Step 1: Add API test for LULC demo job execution**

Append to `tests/test_model_hub_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_model_hub_api.py::test_model_hub_runs_lulc_demo_patch_job -q
```

Expected: fail because `app.api.model_hub` has no `run_model_hub_job` import and job route does not execute runtimes.

- [ ] **Step 3: Implement runtime dispatcher**

Create `ae_backend/app/services/model_hub_runtime.py`:

```python
from __future__ import annotations

import numpy as np


class ModelHubRuntimeError(ValueError):
    """Raised when a model-hub job cannot be executed."""


def _run_lulc_demo_patch(model_id: str, options: dict) -> dict:
    from app.api.inference import get_lulc_service

    service = get_lulc_service(model_id=model_id)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    prediction = service.predict_image(image)
    return {
        "result": {
            "task": "lulc_segmentation",
            "model_id": prediction.get("model_id", model_id),
            "summary": {
                "class_pixel_counts": prediction.get("class_pixel_counts", {}),
                "class_area_fraction": prediction.get("class_area_fraction", {}),
                "mask_shape": prediction.get("mask_shape"),
            },
        },
        "artifacts": [{"kind": "json", "path": "inline"}],
        "logs": ["ran LULC demo patch runtime"],
    }


def run_model_hub_job(*, model_id: str, input_mode: str, options: dict) -> dict:
    if model_id == "lulc_6class_prithvi_houlsby" and input_mode == "demo_patch":
        return _run_lulc_demo_patch(model_id, options)
    raise ModelHubRuntimeError(f"Unsupported model-hub job: model_id={model_id}, input_mode={input_mode}")
```

- [ ] **Step 4: Execute runtime inside job route**

Modify `ae_backend/app/api/model_hub.py`:

```python
from app.services.model_hub_runtime import ModelHubRuntimeError, run_model_hub_job
```

Replace `create_job` with:

```python
@router.post("/jobs")
def create_job(request: ModelHubJobRequest):
    try:
        get_model_registry().get_model(request.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {request.model_id}") from exc

    job = JOB_STORE.create_job(
        model_id=request.model_id,
        input_mode=request.input_mode,
        options=request.options,
    )
    should_execute_now = (
        request.model_id == "lulc_6class_prithvi_houlsby"
        and request.input_mode == "demo_patch"
    )
    if not should_execute_now:
        return JOB_STORE.get_job(job["job_id"])
    try:
        JOB_STORE.mark_running(job["job_id"], log="job accepted")
        runtime_result = run_model_hub_job(
            model_id=request.model_id,
            input_mode=request.input_mode,
            options=request.options,
        )
        JOB_STORE.mark_succeeded(
            job["job_id"],
            result=runtime_result["result"],
            artifacts=runtime_result["artifacts"],
            log=runtime_result["logs"][-1] if runtime_result.get("logs") else "job finished",
        )
    except ModelHubRuntimeError as exc:
        JOB_STORE.mark_failed(job["job_id"], error=str(exc))
    except Exception as exc:
        JOB_STORE.mark_failed(job["job_id"], error=str(exc))
    return JOB_STORE.get_job(job["job_id"])
```

- [ ] **Step 5: Run the new API test**

Run:

```bash
python -m pytest tests/test_model_hub_api.py::test_model_hub_runs_lulc_demo_patch_job -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add ae_backend/app/services/model_hub_runtime.py ae_backend/app/api/model_hub.py tests/test_model_hub_api.py
git commit -m "feat: run model hub demo jobs"
```

## Task 5: Raster Pipeline Utilities

**Files:**
- Create: `ae_backend/app/services/raster_pipeline.py`
- Test: `tests/test_raster_pipeline.py`

- [ ] **Step 1: Write raster utility tests**

Create `tests/test_raster_pipeline.py`:

```python
import sys
from pathlib import Path

import numpy as np


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_make_tile_grid_covers_array_with_overlap():
    from app.services.raster_pipeline import make_tile_grid

    tiles = make_tile_grid(width=300, height=260, tile_size=128, stride=96)

    assert tiles[0] == {"x0": 0, "y0": 0, "x1": 128, "y1": 128}
    assert tiles[-1]["x1"] == 300
    assert tiles[-1]["y1"] == 260
    assert all(tile["x1"] > tile["x0"] and tile["y1"] > tile["y0"] for tile in tiles)


def test_compute_class_area_summary_counts_pixels():
    from app.services.raster_pipeline import compute_class_area_summary

    mask = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)
    summary = compute_class_area_summary(mask, class_names=["background", "built", "water"])

    assert summary["class_pixel_counts"] == {"background": 1, "built": 2, "water": 3}
    assert summary["class_area_fraction"]["water"] == 0.5


def test_stitch_class_tiles_overwrites_expected_window():
    from app.services.raster_pipeline import stitch_class_tiles

    tiles = [
        ({"x0": 0, "y0": 0, "x1": 2, "y1": 2}, np.ones((2, 2), dtype=np.uint8)),
        ({"x0": 1, "y0": 1, "x1": 3, "y1": 3}, np.full((2, 2), 2, dtype=np.uint8)),
    ]
    stitched = stitch_class_tiles(width=3, height=3, tiles=tiles, fill_value=0)

    assert stitched.tolist() == [[1, 1, 0], [1, 2, 2], [0, 2, 2]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_raster_pipeline.py -q
```

Expected: fail with missing `raster_pipeline`.

- [ ] **Step 3: Implement raster utilities**

Create `ae_backend/app/services/raster_pipeline.py`:

```python
from __future__ import annotations

import numpy as np


def _starts(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]
    starts = list(range(0, max(size - tile_size, 0) + 1, stride))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tile_grid(width: int, height: int, tile_size: int, stride: int) -> list[dict[str, int]]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    tiles: list[dict[str, int]] = []
    for y0 in _starts(height, tile_size, stride):
        for x0 in _starts(width, tile_size, stride):
            tiles.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(min(x0 + tile_size, width)),
                    "y1": int(min(y0 + tile_size, height)),
                }
            )
    return tiles


def compute_class_area_summary(mask: np.ndarray, class_names: list[str]) -> dict:
    mask_array = np.asarray(mask)
    total = int(mask_array.size)
    denominator = max(total, 1)
    counts: dict[str, int] = {}
    fractions: dict[str, float] = {}
    for class_id, class_name in enumerate(class_names):
        count = int(np.count_nonzero(mask_array == class_id))
        counts[class_name] = count
        fractions[class_name] = count / denominator
    return {"class_pixel_counts": counts, "class_area_fraction": fractions}


def stitch_class_tiles(
    *,
    width: int,
    height: int,
    tiles: list[tuple[dict[str, int], np.ndarray]],
    fill_value: int = 0,
) -> np.ndarray:
    stitched = np.full((height, width), fill_value, dtype=np.uint8)
    for window, tile_mask in tiles:
        y0, y1 = int(window["y0"]), int(window["y1"])
        x0, x1 = int(window["x0"]), int(window["x1"])
        stitched[y0:y1, x0:x1] = np.asarray(tile_mask, dtype=np.uint8)[: y1 - y0, : x1 - x0]
    return stitched
```

- [ ] **Step 4: Run raster tests**

Run:

```bash
python -m pytest tests/test_raster_pipeline.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add ae_backend/app/services/raster_pipeline.py tests/test_raster_pipeline.py
git commit -m "feat: add model hub raster utilities"
```

## Task 6: Change Detection Productization

**Files:**
- Create: `ae_backend/app/services/model_hub_change.py`
- Modify: `ae_backend/app/services/model_hub_runtime.py`
- Test: `tests/test_model_hub_change.py`
- Test: `tests/test_model_hub_api.py`

- [ ] **Step 1: Write change service tests**

Create `tests/test_model_hub_change.py`:

```python
import json
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_summarize_cached_linhe_change_reads_heatmap(tmp_path: Path):
    from app.services.model_hub_change import summarize_cached_linhe_change

    change_dir = tmp_path / "linhe_change"
    pair_dir = change_dir / "2025Q1_vs_2025Q4"
    pair_dir.mkdir(parents=True)
    heatmap = change_dir / "change_heatmap_2025Q1_vs_2025Q4.geojson"
    heatmap.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [107.1, 40.8]},
                        "properties": {"mean_pca_score": 0.42, "mean_rgb_diff": 0.12, "patch_a": "p_00001_00002.npz"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "pair_visual_00001_00002.png").write_bytes(b"png")

    result = summarize_cached_linhe_change(options={"change_dir": str(change_dir), "top": 1})

    assert result["result"]["task"] == "change_detection"
    assert result["result"]["summary"]["n_features"] == 1
    assert result["result"]["summary"]["top_mean_pca_score"] == 0.42
    artifact_kinds = {artifact["kind"] for artifact in result["artifacts"]}
    assert {"geojson", "png"}.issubset(artifact_kinds)
```

- [ ] **Step 2: Add API test for cached change runtime**

Append to `tests/test_model_hub_api.py`:

```python
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
```

- [ ] **Step 3: Run change tests to verify they fail**

Run:

```bash
python -m pytest tests/test_model_hub_change.py tests/test_model_hub_api.py::test_model_hub_runs_cached_change_job -q
```

Expected: fail with missing `model_hub_change`.

- [ ] **Step 4: Implement change service**

Create `ae_backend/app/services/model_hub_change.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from app.core.config import PROJECT_ROOT


def default_change_dir() -> Path:
    return Path(PROJECT_ROOT) / "results" / "linhe_change"


def summarize_cached_linhe_change(*, options: dict) -> dict:
    change_dir = Path(options.get("change_dir") or default_change_dir())
    top = int(options.get("top", 50))
    heatmap = change_dir / "change_heatmap_2025Q1_vs_2025Q4.geojson"
    pair_dir = change_dir / "2025Q1_vs_2025Q4"
    if not heatmap.exists():
        raise FileNotFoundError(f"Missing change heatmap: {heatmap}")
    feature_collection = json.loads(heatmap.read_text(encoding="utf-8"))
    features = feature_collection.get("features", [])
    sorted_features = sorted(
        features,
        key=lambda feature: feature.get("properties", {}).get("mean_pca_score", 0),
        reverse=True,
    )[:top]
    top_score = 0.0
    if sorted_features:
        top_score = float(sorted_features[0].get("properties", {}).get("mean_pca_score", 0.0))

    artifacts = [{"kind": "geojson", "path": str(heatmap)}]
    if pair_dir.exists():
        for pair_png in sorted(pair_dir.glob("pair_visual_*.png"))[:top]:
            artifacts.append({"kind": "png", "path": str(pair_png)})

    return {
        "result": {
            "task": "change_detection",
            "model_id": "semantic_change_prithvi",
            "summary": {
                "n_features": len(features),
                "returned_features": len(sorted_features),
                "top_mean_pca_score": top_score,
                "method": "PCA-RX visual change plus semantic differencing slot",
            },
            "features": sorted_features,
        },
        "artifacts": artifacts,
        "logs": [f"loaded cached Linhe change artifacts from {change_dir}"],
    }
```

- [ ] **Step 5: Enable change runtime dispatch**

Modify `ae_backend/app/services/model_hub_runtime.py`:

```python
def run_model_hub_job(*, model_id: str, input_mode: str, options: dict) -> dict:
    if model_id == "lulc_6class_prithvi_houlsby" and input_mode == "demo_patch":
        return _run_lulc_demo_patch(model_id, options)
    if model_id == "semantic_change_prithvi" and input_mode == "cached_demo":
        from app.services.model_hub_change import summarize_cached_linhe_change

        return summarize_cached_linhe_change(options=options)
    raise ModelHubRuntimeError(f"Unsupported model-hub job: model_id={model_id}, input_mode={input_mode}")
```

Modify the `should_execute_now` expression in `ae_backend/app/api/model_hub.py`:

```python
should_execute_now = (
    (request.model_id == "lulc_6class_prithvi_houlsby" and request.input_mode == "demo_patch")
    or (request.model_id == "semantic_change_prithvi" and request.input_mode == "cached_demo")
)
```

- [ ] **Step 6: Run change tests**

Run:

```bash
python -m pytest tests/test_model_hub_change.py tests/test_model_hub_api.py::test_model_hub_runs_cached_change_job -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add ae_backend/app/services/model_hub_change.py ae_backend/app/services/model_hub_runtime.py tests/test_model_hub_change.py tests/test_model_hub_api.py
git commit -m "feat: expose cached change detection jobs"
```

## Task 7: Frontend Model Hub Entry

**Files:**
- Modify: `ae_frontend/index.html`
- Test: `tests/test_model_hub_frontend_entry.py`

- [ ] **Step 1: Write frontend contract test**

Create `tests/test_model_hub_frontend_entry.py`:

```python
from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "ae_frontend" / "index.html"


def test_frontend_exposes_model_hub_tab_and_api_actions():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "currentTab = 'modelHub'" in html
    assert "currentTab === 'modelHub'" in html
    assert "妯″瀷涓績" in html
    assert "/api/ae/model-hub/models" in html
    assert "/api/ae/model-hub/jobs" in html
    assert "runModelHubDemo" in html
    assert "modelHubModels" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: fail because the tab does not exist.

- [ ] **Step 3: Add sidebar entry**

Modify the sidebar list in `ae_frontend/index.html`. Add this item after the existing "妯″瀷璧勪骇搴? item:

```html
<li>
    <a href="#" @click.prevent="currentTab = 'modelHub'" :class="currentTab === 'modelHub' ? 'tab-active' : 'tab-inactive'" class="flex items-center space-x-3 px-3 py-2 rounded-md transition-colors duration-200 cursor-pointer">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"></path></svg>
        <span>妯″瀷涓績</span>
    </a>
</li>
```

- [ ] **Step 4: Add Model Hub content panel**

Add this panel near the existing tab panels:

```html
<div v-show="currentTab === 'modelHub'" class="max-w-7xl mx-auto space-y-6">
    <div class="flex justify-between items-center">
        <div>
            <h2 class="text-2xl font-bold text-gray-900">妯″瀷涓績</h2>
            <p class="text-sm text-gray-500 mt-1">鍟嗕笟绾?Prithvi/AlphaEarth 妯″瀷鍖咃細鏌ョ湅妯″瀷瑙勬牸銆佽繍琛屾紨绀轰换鍔°€佷笅杞界粨鏋溿€?/p>
        </div>
        <button class="px-4 py-2 bg-primary text-white rounded-md text-sm hover:bg-blue-600" @click="fetchModelHubModels">鍒锋柊妯″瀷</button>
    </div>
    <div v-if="modelHubStatus.message" :class="modelHubStatus.type === 'error' ? 'bg-red-50 text-red-700 border-red-200' : 'bg-blue-50 text-blue-700 border-blue-200'" class="border rounded-md px-3 py-2 text-sm">
        {{ modelHubStatus.message }}
    </div>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div v-for="model in modelHubModels" :key="model.model_id" class="glass-card p-4">
            <div class="flex items-start justify-between gap-3">
                <div>
                    <h3 class="text-base font-semibold text-gray-900">{{ model.display_name }}</h3>
                    <p class="text-xs text-gray-500 mt-1">{{ model.model_id }} 路 {{ model.task_type }}</p>
                </div>
                <span class="px-2 py-1 rounded text-xs border" :class="model.status === 'ready' ? 'bg-green-50 text-green-700 border-green-200' : 'bg-gray-50 text-gray-600 border-gray-200'">{{ model.status }}</span>
            </div>
            <div class="mt-3 text-sm text-gray-600">
                <div>Backbone: <span class="data-font">{{ model.backbone }}</span></div>
                <div>Adapter: <span class="data-font">{{ model.adapter }}</span></div>
                <div>Classes: <span class="data-font">{{ model.class_schema.join(', ') }}</span></div>
            </div>
            <button class="mt-4 px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50" @click="runModelHubDemo(model)">
                杩愯婕旂ず浠诲姟
            </button>
        </div>
    </div>
    <div v-if="modelHubJob" class="glass-card p-4">
        <h3 class="text-sm font-semibold text-gray-700 mb-2">鏈€杩戜换鍔?/h3>
        <pre class="text-xs bg-gray-900 text-green-300 rounded p-3 overflow-auto max-h-80">{{ JSON.stringify(modelHubJob, null, 2) }}</pre>
    </div>
</div>
```

- [ ] **Step 5: Add Vue state and methods**

Inside Vue `setup`, add these refs near other feature refs:

```javascript
const modelHubModels = ref([]);
const modelHubJob = ref(null);
const modelHubStatus = reactive({ type: 'info', message: '' });
```

Add these methods:

```javascript
const fetchModelHubModels = async () => {
    try {
        const res = await fetch('/api/ae/model-hub/models');
        if (!res.ok) throw new Error(await res.text());
        const body = await res.json();
        modelHubModels.value = body.models || [];
        modelHubStatus.type = 'info';
        modelHubStatus.message = `宸插姞杞?${modelHubModels.value.length} 涓ā鍨嬪寘銆俙;
    } catch (e) {
        console.error('Failed to load model hub models', e);
        modelHubStatus.type = 'error';
        modelHubStatus.message = '鍔犺浇妯″瀷涓績澶辫触锛岃纭鍚庣 /api/ae/model-hub/models 鍙闂€?;
    }
};

const runModelHubDemo = async (model) => {
    const inputMode = model.model_id === 'semantic_change_prithvi' ? 'cached_demo' : 'demo_patch';
    try {
        const res = await fetch('/api/ae/model-hub/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: model.model_id, input_mode: inputMode, options: { output_formats: ['png', 'geojson', 'csv'] } })
        });
        if (!res.ok) throw new Error(await res.text());
        modelHubJob.value = await res.json();
        modelHubStatus.type = 'info';
        modelHubStatus.message = `浠诲姟 ${modelHubJob.value.job_id} 鐘舵€侊細${modelHubJob.value.status}`;
    } catch (e) {
        console.error('Failed to run model hub job', e);
        modelHubStatus.type = 'error';
        modelHubStatus.message = '妯″瀷涓績浠诲姟杩愯澶辫触锛岃鏌ョ湅鍚庣鏃ュ織銆?;
    }
};
```

Return them from `setup`:

```javascript
modelHubModels, modelHubJob, modelHubStatus, fetchModelHubModels, runModelHubDemo,
```

Call `fetchModelHubModels()` from the existing mounted/init block, next to other initial fetches:

```javascript
fetchModelHubModels();
```

- [ ] **Step 6: Run frontend contract test**

Run:

```bash
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: `1 passed`.

- [ ] **Step 7: Commit Task 7**

Run:

```bash
git add ae_frontend/index.html tests/test_model_hub_frontend_entry.py
git commit -m "feat: add model hub frontend entry"
```

## Task 8: End-To-End Verification

**Files:**
- Modify only files already touched if verification exposes an issue.

- [ ] **Step 1: Run focused backend and frontend tests**

Run:

```bash
python -m pytest tests/test_model_hub_registry.py tests/test_model_hub_jobs.py tests/test_model_hub_api.py tests/test_model_hub_change.py tests/test_raster_pipeline.py tests/test_model_hub_frontend_entry.py tests/test_inference_api.py tests/test_lulc_registry.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run existing Paper 12 regression tests**

Run:

```bash
python -m pytest tests/test_paper12_public_dataset_results.py tests/test_paper12_colab_notebooks.py -q
```

Expected: pass. This confirms the commercial Model Hub work did not disturb the Paper 12 evidence files.

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Inspect the final diff**

Run:

```bash
git diff --stat HEAD
git status --short
```

Expected: only planned Model Hub files are modified or added.

- [ ] **Step 5: Commit verification fixes or final checkpoint**

If Step 4 shows uncommitted verification fixes, run:

```bash
git add ae_backend/app ae_frontend/index.html tests
git commit -m "test: verify model hub phase 1"
```

If there are no uncommitted changes, record the passing commands in the final implementation report.

## Spec Coverage Map

- Model Registry: Tasks 1 and 2.
- Unified Job API: Tasks 2 and 3.
- LULC first runnable model: Task 4.
- Shared raster utilities: Task 5.
- Change-detection productization: Task 6.
- Frontend Model Hub page: Task 7.
- Testing and acceptance criteria: Task 8.
- Planned building/road/water entries: Task 1 registry JSON, marked `planned`.

## Execution Notes

- Keep the first implementation local and deterministic.
- Do not add Celery, Redis, PostGIS migrations, cloud storage, or auth in Phase 1.
- Do not claim building, road, or water models are runnable until checkpoints and validation metrics exist.
- For missing weights, keep the API response explicit: the model can be registered but not runnable unless its checkpoint path is available.
- Prefer small commits after each task, matching this plan.
