# Production Model Realization M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Model Hub production-readiness truthful, add real asset/data evidence, and prepare crop/flood/LULC for checkpoint-backed production inference without mislabeling unfinished models.

**Architecture:** Add a small asset/evidence registry beside the existing model registry, then feed it into capabilities, verification, evidence drill-down, Model Hub job dispatch, and the frontend. Real crop/flood runtimes are optional-dependency adapters: they run only when public weights, sample data, and runtime packages are present; otherwise they fail with explicit dependency/download status.

**Tech Stack:** FastAPI, Pydantic-style dict contracts already used in the app, rasterio, numpy, torch, pytest, optional Hugging Face/TerraTorch runtime hooks, static HTML/JS frontend.

---

## File Structure

- Create `ae_backend/app/data/model_hub_assets.json`: public weights, datasets, local cache paths, licenses, runtime kinds, and readiness gates.
- Create `ae_backend/app/services/model_asset_registry.py`: load/validate asset registry and resolve local file evidence.
- Create `ae_backend/app/services/model_hub_evidence.py`: combine model registry and asset registry into per-model production evidence.
- Modify `ae_backend/app/services/system_capabilities.py`: include production evidence and new readiness counts.
- Modify `ae_backend/app/services/system_verification.py`: add checkpoint/data/dependency gates for production states.
- Modify `ae_backend/app/services/system_evidence.py`: expose asset evidence drill-down using existing safe-preview rules.
- Modify `ae_backend/app/api/model_hub.py`: reject impossible real inference modes immediately with actionable errors.
- Modify `ae_backend/app/services/model_hub_runtime.py`: add dispatch stubs for `real_raster_inference` crop/flood modes guarded by evidence.
- Create `ae_backend/app/services/model_hub_lulc_raster.py`: production-style LULC raster tiling/export wrapper around existing checkpoints.
- Create `ae_backend/app/services/model_hub_real_crop.py`: real crop runtime facade with dependency/weight guards and artifact contract.
- Create `ae_backend/app/services/model_hub_flood.py`: real flood runtime facade with dependency/weight guards and artifact contract.
- Create `scripts/model_hub/verify_assets.py`: offline asset/evidence verifier.
- Create `scripts/model_hub/fetch_public_sample.py`: explicit downloader entrypoint, dry-run first.
- Create tests:
  - `tests/test_model_asset_registry.py`
  - `tests/test_model_hub_evidence.py`
  - `tests/test_model_hub_lulc_raster.py`
  - `tests/test_model_hub_real_runtime_guards.py`
  - extend `tests/test_model_hub_api.py`
  - extend `tests/test_model_hub_frontend_entry.py`

## Task 1: Asset Registry Contract

**Files:**
- Create: `ae_backend/app/data/model_hub_assets.json`
- Create: `ae_backend/app/services/model_asset_registry.py`
- Test: `tests/test_model_asset_registry.py`

- [ ] **Step 1: Write failing tests for asset registry loading**

Create `tests/test_model_asset_registry.py`:

```python
import json
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
    assert building["test_data"]["dataset_id"] in {"spacenet_buildings", "microsoft_building_footprints"}


def test_model_asset_registry_reports_local_file_presence(tmp_path: Path):
    from app.services.model_asset_registry import build_asset_presence

    root = tmp_path
    existing = root / "data" / "weights" / "x.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"weights")

    record = {
        "model_id": "x",
        "weights": {"local_paths": ["data/weights/x.pt", "data/weights/missing.pt"]},
        "test_data": {"local_paths": ["results/missing.tif"]},
    }

    presence = build_asset_presence(record, project_root=root)
    assert presence["weights"]["available"] is False
    assert presence["weights"]["files"][0]["exists"] is True
    assert presence["weights"]["files"][1]["exists"] is False
    assert presence["test_data"]["available"] is False
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_model_asset_registry.py -q
```

Expected: import failure for `app.services.model_asset_registry`.

- [ ] **Step 3: Add asset registry JSON**

Create `ae_backend/app/data/model_hub_assets.json` with the model ids already in `model_hub_models.json`. Include:

```json
{
  "version": 1,
  "models": [
    {
      "model_id": "prithvi_crop_classification_arcgis_style",
      "runtime_kind": "neural_checkpoint",
      "weights": {
        "source": "huggingface",
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification",
        "license": "Apache-2.0",
        "local_paths": ["data/weights/prithvi_crop"],
        "required": true
      },
      "test_data": {
        "dataset_id": "ibm_nasa_multi_temporal_crop",
        "source_url": "https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification",
        "license": "CC-BY-4.0",
        "input_profile": "18_band_hls_multitemporal_composite",
        "local_paths": ["data/public_samples/prithvi_crop"],
        "required_for_ready": true
      },
      "promotion_policy": "ready_requires_weights_data_and_runtime_verification"
    },
    {
      "model_id": "water_flood_prithvi",
      "runtime_kind": "neural_checkpoint",
      "weights": {
        "source": "huggingface",
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11",
        "license": "Apache-2.0",
        "local_paths": ["data/weights/prithvi_flood"],
        "required": true
      },
      "test_data": {
        "dataset_id": "sen1floods11",
        "source_url": "https://github.com/cloudtostreet/Sen1Floods11",
        "license": "public_research_dataset",
        "input_profile": "6_band_sentinel2_flood",
        "local_paths": ["data/sen1floods11"],
        "required_for_ready": true
      },
      "promotion_policy": "ready_requires_weights_data_and_runtime_verification"
    },
    {
      "model_id": "lulc_6class_prithvi_houlsby",
      "runtime_kind": "neural_checkpoint",
      "weights": {
        "source": "local_training",
        "license": "research_demo",
        "local_paths": ["data/weights/linhe_lulc/houlsby__rgb_3band__seed123.pt"],
        "required": true
      },
      "test_data": {
        "dataset_id": "linhe_lulc_local",
        "license": "local_research",
        "input_profile": "rgb_3band_patch_or_raster",
        "local_paths": ["results/model_hub/lulc_inputs/linhe_npz_rgb_patch.png"],
        "required_for_ready": true
      },
      "promotion_policy": "ready_requires_existing_checkpoint_and_smoke_sample"
    },
    {
      "model_id": "building_extraction_prithvi",
      "runtime_kind": "training_pipeline",
      "weights": {"source": "not_trained", "local_paths": [], "required": true},
      "test_data": {
        "dataset_id": "spacenet_buildings",
        "source_url": "https://registry.opendata.aws/spacenet/",
        "license": "SpaceNet terms",
        "input_profile": "rgb_high_resolution_building_masks",
        "local_paths": ["data/public_samples/spacenet_buildings"],
        "required_for_ready": true
      },
      "promotion_policy": "training_required_before_ready"
    },
    {
      "model_id": "road_hardscape_prithvi",
      "runtime_kind": "training_pipeline",
      "weights": {"source": "not_trained", "local_paths": [], "required": true},
      "test_data": {
        "dataset_id": "spacenet_roads",
        "source_url": "https://registry.opendata.aws/spacenet/",
        "license": "SpaceNet terms",
        "input_profile": "rgb_high_resolution_road_masks",
        "local_paths": ["data/public_samples/spacenet_roads"],
        "required_for_ready": true
      },
      "promotion_policy": "training_required_before_ready"
    },
    {
      "model_id": "semantic_change_prithvi",
      "runtime_kind": "training_pipeline",
      "weights": {"source": "not_trained", "local_paths": [], "required": true},
      "test_data": {
        "dataset_id": "spacenet7_change",
        "source_url": "https://registry.opendata.aws/spacenet/",
        "license": "SpaceNet terms",
        "input_profile": "two_date_georegistered_imagery_and_change_labels",
        "local_paths": ["data/public_samples/spacenet7"],
        "required_for_ready": true
      },
      "promotion_policy": "training_required_before_ready"
    }
  ]
}
```

- [ ] **Step 4: Implement registry loader**

Create `ae_backend/app/services/model_asset_registry.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT


DEFAULT_ASSET_REGISTRY_PATH = (
    Path(PROJECT_ROOT) / "ae_backend" / "app" / "data" / "model_hub_assets.json"
)


class ModelAssetRegistryError(ValueError):
    """Raised when model asset metadata is invalid."""


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelAssetRegistryError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelAssetRegistryError(f"{label} must be a non-empty string")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelAssetRegistryError(f"{label} must be a list")
    return value


def _validate_model(record: Any) -> dict[str, Any]:
    item = _require_mapping(record, "asset model")
    _require_string(item.get("model_id"), "model_id")
    _require_string(item.get("runtime_kind"), "runtime_kind")
    _require_mapping(item.get("weights"), "weights")
    _require_mapping(item.get("test_data"), "test_data")
    _require_string(item.get("promotion_policy"), "promotion_policy")
    _require_list(item["weights"].get("local_paths", []), "weights.local_paths")
    _require_list(item["test_data"].get("local_paths", []), "test_data.local_paths")
    return dict(item)


def load_model_asset_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path or DEFAULT_ASSET_REGISTRY_PATH)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    root = _require_mapping(payload, "asset registry")
    models = [_validate_model(record) for record in _require_list(root.get("models"), "models")]
    model_ids = [model["model_id"] for model in models]
    if len(model_ids) != len(set(model_ids)):
        raise ModelAssetRegistryError("model_id values must be unique")
    return {"version": root.get("version", 1), "models": models}


def _file_presence(local_paths: list[str], project_root: Path) -> dict[str, Any]:
    files = []
    for ref in local_paths:
        path = (project_root / ref).resolve()
        files.append({"path": str(ref), "exists": path.exists(), "is_dir": path.is_dir()})
    available = bool(files) and all(item["exists"] for item in files)
    return {"available": available, "files": files}


def build_asset_presence(record: dict[str, Any], *, project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or PROJECT_ROOT)
    weights = record.get("weights", {})
    test_data = record.get("test_data", {})
    return {
        "model_id": record["model_id"],
        "runtime_kind": record["runtime_kind"],
        "weights": _file_presence(list(weights.get("local_paths", [])), root),
        "test_data": _file_presence(list(test_data.get("local_paths", [])), root),
    }
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_model_asset_registry.py -q
```

Expected: `2 passed`.

## Task 2: Production Evidence Service

**Files:**
- Create: `ae_backend/app/services/model_hub_evidence.py`
- Modify: `ae_backend/app/services/system_capabilities.py`
- Modify: `ae_backend/app/services/system_verification.py`
- Test: `tests/test_model_hub_evidence.py`

- [ ] **Step 1: Write failing evidence tests**

Create `tests/test_model_hub_evidence.py`:

```python
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
    assert crop["production_state"] in {"download_required", "dependency_required", "verification_required"}
    assert crop["weights"]["source"] == "huggingface"
    assert crop["may_run_real_inference"] is False


def test_model_hub_evidence_marks_lulc_checkpoint_available():
    from app.api.model_hub import get_model_registry
    from app.services.model_hub_evidence import build_model_hub_evidence

    evidence = build_model_hub_evidence(get_model_registry())
    lulc = {item["model_id"]: item for item in evidence["models"]}["lulc_6class_prithvi_houlsby"]

    assert lulc["runtime_kind"] == "neural_checkpoint"
    assert lulc["weights"]["presence"]["available"] is True
    assert lulc["production_state"] in {"verification_required", "production_candidate"}
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/test_model_hub_evidence.py -q
```

Expected: import failure for `model_hub_evidence`.

- [ ] **Step 3: Implement evidence service**

Create `ae_backend/app/services/model_hub_evidence.py` with:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services.model_asset_registry import build_asset_presence, load_model_asset_registry
from app.services.model_hub_registry import ModelHubRegistry


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _state(model: dict[str, Any], asset: dict[str, Any] | None, presence: dict[str, Any] | None) -> str:
    if asset is None or presence is None:
        return "metadata_missing"
    if asset["runtime_kind"] == "training_pipeline":
        return "training_required"
    if not presence["weights"]["available"]:
        return "download_required"
    if not presence["test_data"]["available"]:
        return "test_data_required"
    if model.get("status") == "ready":
        return "production_candidate"
    return "verification_required"


def build_model_hub_evidence(
    registry: ModelHubRegistry,
    *,
    asset_registry: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    assets = asset_registry or load_model_asset_registry()
    by_id = {item["model_id"]: item for item in assets["models"]}
    root = Path(project_root or PROJECT_ROOT)
    models = []
    for model_entry in registry.models:
        model = model_entry.to_dict()
        asset = by_id.get(model["model_id"])
        presence = build_asset_presence(asset, project_root=root) if asset else None
        production_state = _state(model, asset, presence)
        may_run = production_state in {"production_candidate", "verification_required"} and asset and asset["runtime_kind"] == "neural_checkpoint"
        models.append(
            {
                "model_id": model["model_id"],
                "registry_status": model.get("status"),
                "runtime_kind": asset["runtime_kind"] if asset else "metadata_missing",
                "production_state": production_state,
                "may_run_real_inference": bool(may_run and presence and presence["weights"]["available"]),
                "weights": {
                    **(asset.get("weights", {}) if asset else {}),
                    "presence": presence["weights"] if presence else {"available": False, "files": []},
                },
                "test_data": {
                    **(asset.get("test_data", {}) if asset else {}),
                    "presence": presence["test_data"] if presence else {"available": False, "files": []},
                },
                "promotion_policy": asset.get("promotion_policy") if asset else "asset_metadata_required",
            }
        )
    return {"generated_at": _utc_now(), "models": models}
```

- [ ] **Step 4: Integrate into capabilities and verification**

Modify `system_capabilities.build_system_capabilities()` to attach evidence by model id:

```python
from app.services.model_hub_evidence import build_model_hub_evidence

production_evidence = build_model_hub_evidence(registry)
evidence_by_id = {item["model_id"]: item for item in production_evidence["models"]}
capabilities = [_capability(model) for model in models]
for capability in capabilities:
    capability["production_evidence"] = evidence_by_id.get(
        capability["id"],
        {
            "model_id": capability["id"],
            "runtime_kind": "metadata_missing",
            "production_state": "metadata_missing",
            "may_run_real_inference": False,
        },
    )
```

Modify `system_verification.build_system_verification()` to add a check:

```python
def _production_evidence_check(capability: dict[str, Any]) -> dict[str, Any]:
    evidence = capability.get("production_evidence") or {}
    state = str(evidence.get("production_state") or "metadata_missing")
    capability_id = capability["id"]
    if state in {"production_candidate", "verification_required"}:
        return _check(
            capability_id=capability_id,
            check_id="production_evidence",
            category="production_evidence",
            status="pass",
            severity="info",
            title="Production evidence is locally usable",
            detail=f"Production evidence state is {state}.",
            evidence_refs=["system_capabilities"],
        )
    if state in {"download_required", "test_data_required", "training_required"}:
        return _check(
            capability_id=capability_id,
            check_id="production_evidence",
            category="production_evidence",
            status="warning",
            severity="warning",
            title="Production evidence is not complete",
            detail=f"Production evidence state is {state}.",
            evidence_refs=["system_capabilities"],
            remediation="Attach the required weights, test data, or training output before promoting this model.",
        )
    return _check(
        capability_id=capability_id,
        check_id="production_evidence",
        category="production_evidence",
        status="fail",
        severity="error",
        title="Production evidence metadata is missing",
        detail=f"Production evidence state is {state}.",
        evidence_refs=["system_capabilities"],
        remediation="Add this model to ae_backend/app/data/model_hub_assets.json.",
    )
```

Append `_production_evidence_check(capability)` to `capability_checks`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_model_hub_evidence.py tests/test_model_hub_api.py -q
```

Expected: existing system endpoint tests still pass, with new production evidence present.

## Task 3: Real-Inference Job Guards

**Files:**
- Modify: `ae_backend/app/api/model_hub.py`
- Modify: `ae_backend/app/services/model_hub_runtime.py`
- Test: extend `tests/test_model_hub_api.py`

- [ ] **Step 1: Write failing API tests**

Append to `tests/test_model_hub_api.py`:

```python
def test_model_hub_rejects_crop_real_inference_when_assets_missing():
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "real_raster_inference",
            "options": {"raster_path": "results/model_hub/prithvi_crop_inputs/crop_18band_demo.tif"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "download" in body["error"].lower() or "dependency" in body["error"].lower()


def test_model_hub_rejects_flood_real_inference_when_assets_missing():
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "water_flood_prithvi",
            "input_mode": "real_raster_inference",
            "options": {"raster_path": "data/sen1floods11/sample.tif"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "water_flood_prithvi" in body["error"]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_model_hub_rejects_crop_real_inference_when_assets_missing tests/test_model_hub_api.py::test_model_hub_rejects_flood_real_inference_when_assets_missing -q
```

Expected: jobs remain pending or unsupported before guard implementation.

- [ ] **Step 3: Implement guard**

In `ae_backend/app/api/model_hub.py`, before creating a job for `real_raster_inference`, call evidence:

```python
from app.services.model_hub_evidence import build_model_hub_evidence


def _real_inference_guard(model_id: str, input_mode: str) -> str | None:
    if input_mode != "real_raster_inference":
        return None
    evidence = build_model_hub_evidence(get_model_registry())
    by_id = {item["model_id"]: item for item in evidence["models"]}
    model = by_id.get(model_id)
    if not model:
        return f"{model_id} has no production asset evidence configured."
    if not model["may_run_real_inference"]:
        return (
            f"{model_id} cannot run real inference yet: "
            f"production_state={model['production_state']}."
        )
    return None
```

Use it after model lookup and before runtime dispatch. If blocked, create the job, mark it failed, and return it.

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py -q
```

Expected: Model Hub API tests pass.

## Task 4: LULC Raster Production Wrapper

**Files:**
- Create: `ae_backend/app/services/model_hub_lulc_raster.py`
- Modify: `ae_backend/app/services/model_hub_runtime.py`
- Test: `tests/test_model_hub_lulc_raster.py`

- [ ] **Step 1: Write failing raster wrapper test**

Create `tests/test_model_hub_lulc_raster.py`:

```python
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def _write_rgb_geotiff(path: Path) -> Path:
    data = np.zeros((3, 16, 16), dtype=np.uint8)
    data[0] = 120
    data[1] = 90
    data[2] = 60
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(107.0, 41.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)
    return path


def test_lulc_raster_runtime_writes_gis_artifacts(tmp_path: Path, monkeypatch):
    import app.services.model_hub_lulc_raster as service

    raster_path = _write_rgb_geotiff(tmp_path / "rgb.tif")
    output_dir = tmp_path / "out"

    def fake_predict_image(image):
        h, w, _ = image.shape
        mask = np.full((h, w), 2, dtype=np.int64)
        return {
            "model_id": "fake-lulc",
            "mask": mask.tolist(),
            "classes": ["background", "built", "crops", "trees", "water", "rangeland_bare"],
            "class_pixel_counts": {"crops": int(mask.size)},
            "class_area_fraction": {"crops": 1.0},
            "mask_shape": [h, w],
        }

    monkeypatch.setattr(service, "_predict_rgb_tile", lambda image, model_id, checkpoint_path: fake_predict_image(image))

    result = service.run_lulc_raster_inference(
        options={
            "raster_path": str(raster_path),
            "output_dir": str(output_dir),
            "tile_size": 8,
            "stride": 8,
        }
    )

    assert result["result"]["task"] == "lulc_segmentation"
    kinds = {artifact["kind"] for artifact in result["artifacts"]}
    assert {"geotiff", "csv", "geojson", "manifest", "png"}.issubset(kinds)
    with rasterio.open(output_dir / "classified_lulc.tif") as classified:
        assert classified.width == 16
        assert classified.height == 16
        assert classified.crs.to_string() == "EPSG:4326"
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
python -m pytest tests/test_model_hub_lulc_raster.py -q
```

Expected: import failure for `model_hub_lulc_raster`.

- [ ] **Step 3: Implement raster wrapper**

Implement `run_lulc_raster_inference(options)` by reusing the raster helper patterns from `model_hub_crop_raster.py`: validate 3-band GeoTIFF, tile windows, call `_predict_rgb_tile`, stitch masks, write `classified_lulc.tif`, `lulc_summary.csv`, `lulc_polygons.geojson`, `manifest.json`, and `lulc_preview.png`.

Use this internal seam:

```python
def _predict_rgb_tile(image: np.ndarray, model_id: str | None, checkpoint_path: str | None) -> dict:
    from app.api.inference import get_lulc_service
    service = get_lulc_service(checkpoint_path=checkpoint_path, model_id=model_id)
    return service.predict_image(image)
```

- [ ] **Step 4: Dispatch from Model Hub**

In `model_hub_runtime.py`, add:

```python
if model_id == "lulc_6class_prithvi_houlsby" and input_mode == "raster_inference":
    from app.services.model_hub_lulc_raster import run_lulc_raster_inference
    return run_lulc_raster_inference(options=options)
```

In `model_hub.py`, treat `raster_inference` as executable for LULC.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_model_hub_lulc_raster.py tests/test_model_hub_api.py tests/test_inference_api.py -q
```

Expected: all pass.

## Task 5: Real Crop/Flood Runtime Facades

**Files:**
- Create: `ae_backend/app/services/model_hub_real_crop.py`
- Create: `ae_backend/app/services/model_hub_flood.py`
- Modify: `ae_backend/app/services/model_hub_runtime.py`
- Test: `tests/test_model_hub_real_runtime_guards.py`

- [ ] **Step 1: Write failing guard tests**

Create `tests/test_model_hub_real_runtime_guards.py`:

```python
import sys
from pathlib import Path

import numpy as np
import rasterio
import pytest
from rasterio.transform import from_origin

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def _write_geotiff(path: Path, bands: int) -> Path:
    data = np.zeros((bands, 8, 8), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(100.0, 40.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)
    return path


def test_real_crop_runtime_rejects_missing_weights(tmp_path: Path):
    from app.services.model_hub_real_crop import run_real_crop_inference
    from app.services.model_hub_runtime import ModelHubRuntimeError

    raster = _write_geotiff(tmp_path / "crop.tif", bands=18)
    with pytest.raises(ModelHubRuntimeError, match="crop.*weights|Prithvi.*weights"):
        run_real_crop_inference(options={"raster_path": str(raster), "weights_dir": str(tmp_path / "missing")})


def test_real_flood_runtime_validates_six_bands(tmp_path: Path):
    from app.services.model_hub_flood import validate_flood_raster

    raster = _write_geotiff(tmp_path / "flood.tif", bands=6)
    validation = validate_flood_raster(raster)
    assert validation["band_count"] == 6
    assert validation["width"] == 8


def test_real_flood_runtime_rejects_wrong_band_count(tmp_path: Path):
    from app.services.model_hub_flood import validate_flood_raster
    from app.services.model_hub_runtime import ModelHubRuntimeError

    raster = _write_geotiff(tmp_path / "flood_bad.tif", bands=3)
    with pytest.raises(ModelHubRuntimeError, match="requires 6 bands"):
        validate_flood_raster(raster)
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/test_model_hub_real_runtime_guards.py -q
```

Expected: import failures.

- [ ] **Step 3: Implement crop facade**

Create `model_hub_real_crop.py` with:

```python
from __future__ import annotations

from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.services.model_hub_crop_raster import validate_prithvi_crop_raster
from app.services.model_hub_runtime import ModelHubRuntimeError


def _default_weights_dir() -> Path:
    return Path(PROJECT_ROOT) / "data" / "weights" / "prithvi_crop"


def _require_crop_runtime(weights_dir: Path) -> None:
    if not weights_dir.exists():
        raise ModelHubRuntimeError(
            f"Prithvi crop weights are missing at {weights_dir}. "
            "Run scripts/model_hub/fetch_public_sample.py --asset prithvi_crop --include-weights after approving network download."
        )
    try:
        import terratorch  # noqa: F401
    except ImportError as exc:
        raise ModelHubRuntimeError(
            "Prithvi crop real inference requires TerraTorch-compatible runtime dependencies."
        ) from exc


def run_real_crop_inference(*, options: dict) -> dict:
    raster_path = options.get("raster_path")
    if not raster_path:
        raise ModelHubRuntimeError("raster_path is required for real crop inference")
    validation = validate_prithvi_crop_raster(raster_path)
    weights_dir = Path(options.get("weights_dir") or _default_weights_dir())
    _require_crop_runtime(weights_dir)
    raise ModelHubRuntimeError(
        "Prithvi crop neural runtime is configured but the TerraTorch inference adapter has not completed local verification."
    )
```

This is a production guard, not a fake implementation. It prevents demo fallback and gives the exact install/download blocker.

- [ ] **Step 4: Implement flood facade**

Create `model_hub_flood.py` with a six-band validator, default weights dir, dependency guard, and the same explicit failure if runtime deps/weights are absent.

- [ ] **Step 5: Wire runtime dispatch**

In `model_hub_runtime.py`:

```python
if model_id == "prithvi_crop_classification_arcgis_style" and input_mode == "real_raster_inference":
    from app.services.model_hub_real_crop import run_real_crop_inference
    return run_real_crop_inference(options=options)
if model_id == "water_flood_prithvi" and input_mode == "real_raster_inference":
    from app.services.model_hub_flood import run_real_flood_inference
    return run_real_flood_inference(options=options)
```

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m pytest tests/test_model_hub_real_runtime_guards.py tests/test_model_hub_api.py -q
```

Expected: all pass and no deterministic fallback for real modes.

## Task 6: Explicit Public Sample Downloader Dry Run

**Files:**
- Create: `scripts/model_hub/fetch_public_sample.py`
- Create: `scripts/model_hub/verify_assets.py`
- Test: `tests/test_model_hub_asset_scripts.py`

- [ ] **Step 1: Write script tests**

Create `tests/test_model_hub_asset_scripts.py`:

```python
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


def test_fetch_public_sample_dry_run_lists_crop_source():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/model_hub/fetch_public_sample.py",
            "--asset",
            "prithvi_crop",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "ibm-nasa-geospatial" in result.stdout
    assert "dry-run" in result.stdout.lower()


def test_verify_assets_outputs_json():
    result = subprocess.run(
        [sys.executable, "scripts/model_hub/verify_assets.py", "--json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "prithvi_crop_classification_arcgis_style" in result.stdout
```

- [ ] **Step 2: Implement dry-run scripts**

`fetch_public_sample.py` should print exact source repo/URL and local target. It must not download unless `--execute` is passed.

`verify_assets.py` should load `build_model_hub_evidence()` and print JSON or table output.

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m pytest tests/test_model_hub_asset_scripts.py -q
```

Expected: dry-run tests pass without network.

## Task 7: Frontend Production State Display

**Files:**
- Modify: `ae_frontend/index.html`
- Test: extend `tests/test_model_hub_frontend_entry.py`

- [ ] **Step 1: Write failing frontend tests**

Append assertions:

```python
def test_frontend_exposes_production_readiness_labels():
    html = FRONTEND.read_text(encoding="utf-8")
    assert "鐢熶骇鍊欓€? in html
    assert "闇€瑕佷笅杞? in html
    assert "闇€瑕佽缁? in html
    assert "鐪熷疄鎺ㄧ悊" in html
    assert "濂戠害婕旂ず" in html


def test_frontend_warns_when_model_uses_demo_or_cached_outputs():
    html = FRONTEND.read_text(encoding="utf-8")
    assert "涓嶈灏嗘紨绀虹粨鏋滆В閲婁负鐢熶骇鎺ㄧ悊" in html
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: new labels missing.

- [ ] **Step 3: Update UI**

In the Model Hub render logic, map `production_evidence.production_state` to Chinese labels:

```javascript
const productionStateLabels = {
  production_candidate: '鐢熶骇鍊欓€?,
  verification_required: '闇€瑕侀獙璇?,
  download_required: '闇€瑕佷笅杞?,
  test_data_required: '闇€瑕佹祴璇曟暟鎹?,
  dependency_required: '闇€瑕佷緷璧?,
  training_required: '闇€瑕佽缁?,
  metadata_missing: '缂哄皯鍏冩暟鎹?
};
```

Render runtime kind:

```javascript
const runtimeKindLabels = {
  neural_checkpoint: '鐪熷疄鎺ㄧ悊',
  public_product: '鍏叡浜у搧',
  cached_artifact: '缂撳瓨缁撴灉',
  training_pipeline: '璁粌绠＄嚎',
  contract_demo: '濂戠害婕旂ず'
};
```

Add the warning text for cached/demo modes.

- [ ] **Step 4: Run frontend tests**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: pass.

## Task 8: Verification Pass

**Files:**
- All modified files

- [ ] **Step 1: Run focused test suite**

Run:

```powershell
python -m pytest tests/test_model_asset_registry.py tests/test_model_hub_evidence.py tests/test_model_hub_real_runtime_guards.py tests/test_model_hub_lulc_raster.py tests/test_model_hub_api.py tests/test_model_hub_frontend_entry.py -q
```

Expected: all pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all existing tests pass; document any known environment-level Torch access violation if exit code remains 0.

- [ ] **Step 3: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors from files changed in this work.

- [ ] **Step 4: Report implementation status**

Report:

- which models have real checkpoint evidence;
- which models are blocked by missing downloads/dependencies;
- exact public sample paths available locally;
- exact commands needed to approve/download public data and weights.

## Execution Notes

- Do not use network in unit tests.
- Do not commit large weights or datasets.
- Do not mark crop/flood as `ready` until the real public checkpoint runtime has executed locally.
- Do not silently run `upload_raster_demo` when the user requested `real_raster_inference`.
- Keep unrelated dirty files untouched.

