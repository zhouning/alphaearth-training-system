# Prithvi Crop Model Hub Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ArcGIS-style Prithvi crop-classification package to the existing Paper 12 Model Hub, with registry metadata, a deterministic cached demo runtime, API job dispatch, and frontend demo-mode selection from model metadata.

**Architecture:** Extend the registry to preserve optional model-package metadata, add one `demo_only` crop model entry, implement a small cached runtime under `app.services.model_hub_crop`, and wire it through the existing synchronous Model Hub job path. Keep the implementation local and deterministic; no Prithvi weight download, GPU inference, ArcPy dependency, or `.dlpk` export is part of this phase.

**Tech Stack:** Python 3, FastAPI, Pydantic, dataclasses, JSON registry, pathlib, pytest, FastAPI TestClient, existing Vue-in-HTML frontend.

---

## Source Spec

- Spec: `docs/superpowers/specs/2026-06-27-prithvi-crop-model-hub-phase2-design.md`
- Worktree: `D:\tmp\alphaearth-paper12-results-20260619`
- Branch: `paper12-results-colab-20260619`
- Current base commit: `1065bda docs: add prithvi crop model hub phase 2 design`

## File Structure

Modify these backend files:

- `ae_backend/app/services/model_hub_registry.py`
  Preserve optional registry fields such as `package_profile` in `ModelHubEntry.to_dict()`.

- `ae_backend/app/data/model_hub_models.json`
  Add `prithvi_crop_classification_arcgis_style` with ArcGIS-style package metadata and `status=demo_only`.

- `ae_backend/app/services/model_hub_runtime.py`
  Dispatch the crop model's `cached_demo` jobs to the new crop runtime.

- `ae_backend/app/api/model_hub.py`
  Treat the crop model's `cached_demo` request as a synchronous runnable demo job.

- `ae_frontend/index.html`
  Make Model Hub runnable checks and input-mode selection read runtime metadata instead of only hard-coded model ids.

Create these files:

- `ae_backend/app/services/model_hub_crop.py`
  Deterministic cached crop-classification summary and artifact manifest service.

- `tests/test_model_hub_crop.py`
  Unit tests for crop runtime behavior.

Modify these tests:

- `tests/test_model_hub_registry.py`
  Add tests for optional metadata passthrough and committed crop registry entry.

- `tests/test_model_hub_api.py`
  Add tests for crop model detail and crop demo job execution.

- `tests/test_model_hub_frontend_entry.py`
  Add tests for metadata-driven demo input-mode selection.

## Task 1: Registry Metadata Passthrough And Crop Entry

**Files:**
- Modify: `ae_backend/app/services/model_hub_registry.py`
- Modify: `ae_backend/app/data/model_hub_models.json`
- Modify: `tests/test_model_hub_registry.py`

- [ ] **Step 1: Write failing registry tests**

Append these tests to `tests/test_model_hub_registry.py`:

```python
def test_load_model_registry_preserves_optional_package_profile(tmp_path: Path):
    registry_path = tmp_path / "model_hub_models.json"
    record = _model_record("prithvi_crop_classification_arcgis_style", task_type="crop_classification", status="demo_only")
    record["package_profile"] = {
        "package_type": "arcgis_style_pretrained_imagery_model",
        "runtime_modes": ["cached_demo"],
        "input_profile": {"raster_profile": "multiband_crop_composite"},
    }
    _write_registry(registry_path, [record])

    registry = load_model_registry(registry_path)
    payload = registry.get_model("prithvi_crop_classification_arcgis_style").to_dict()

    assert payload["package_profile"]["package_type"] == "arcgis_style_pretrained_imagery_model"
    assert payload["package_profile"]["runtime_modes"] == ["cached_demo"]
    assert payload["package_profile"]["input_profile"]["raster_profile"] == "multiband_crop_composite"


def test_committed_model_hub_registry_loads_prithvi_crop_package():
    registry = load_model_registry(REGISTRY_DATA_PATH)
    crop = registry.get_model("prithvi_crop_classification_arcgis_style").to_dict()

    assert crop["task_type"] == "crop_classification"
    assert crop["status"] == "demo_only"
    assert crop["input_spec"]["default_demo_input_mode"] == "cached_demo"
    assert "maize" in crop["class_schema"]
    assert crop["package_profile"]["family"] == "prithvi_crop_classification"
    assert crop["package_profile"]["runtime_modes"] == ["cached_demo"]
    assert crop["package_profile"]["applicability"]["readiness"] == "demo_contract_only"
```

Update the existing `test_committed_model_hub_registry_loads_phase_1_models` expected statuses to include the new model:

```python
    assert statuses == {
        "lulc_6class_prithvi_houlsby": "ready",
        "building_extraction_prithvi": "planned",
        "road_hardscape_prithvi": "planned",
        "water_flood_prithvi": "planned",
        "semantic_change_prithvi": "demo_only",
        "prithvi_crop_classification_arcgis_style": "demo_only",
    }
```

- [ ] **Step 2: Run registry tests to verify failure**

Run:

```bash
python -m pytest tests/test_model_hub_registry.py -q
```

Expected: fail because `package_profile` is not preserved by `ModelHubEntry.to_dict()` and the committed registry does not yet include `prithvi_crop_classification_arcgis_style`.

- [ ] **Step 3: Preserve optional metadata in registry entries**

Modify `ae_backend/app/services/model_hub_registry.py`.

Add this import:

```python
from copy import deepcopy
```

Add `extra_fields` to the dataclass:

```python
    extra_fields: dict[str, Any]
```

Inside `ModelHubEntry.from_record`, after the required fields are read and before the dataclass constructor call, add:

```python
        extra_fields = {
            key: deepcopy(value)
            for key, value in record.items()
            if key not in REQUIRED_FIELDS
        }
```

Pass it to the dataclass constructor:

```python
            extra_fields=extra_fields,
```

Replace `to_dict()` with:

```python
    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "backbone": self.backbone,
            "adapter": self.adapter,
            "checkpoint_path": self.checkpoint_path,
            "input_spec": deepcopy(self.input_spec),
            "output_spec": deepcopy(self.output_spec),
            "class_schema": list(self.class_schema),
            "metrics": deepcopy(self.metrics),
            "trained_region": self.trained_region,
            "supported_sensors": list(self.supported_sensors),
            "license": self.license,
            "status": self.status,
            "example_inputs": list(self.example_inputs),
        }
        payload.update(deepcopy(self.extra_fields))
        return payload
```

- [ ] **Step 4: Add crop package registry entry**

Insert this object in `ae_backend/app/data/model_hub_models.json` before `semantic_change_prithvi`:

```json
  {
    "model_id": "prithvi_crop_classification_arcgis_style",
    "display_name": "Prithvi Crop Classification Package",
    "task_type": "crop_classification",
    "backbone": "Prithvi-100M",
    "adapter": "crop classification head contract",
    "checkpoint_path": null,
    "input_spec": {
      "modalities": ["multiband_composite"],
      "bands": "multiband_crop_composite",
      "patch_size": "variable",
      "default_demo_input_mode": "cached_demo",
      "normalization": "model_package_defined"
    },
    "output_spec": {
      "type": "categorical_crop_raster",
      "classes": 13,
      "formats": ["png", "geojson", "csv"]
    },
    "class_schema": [
      "background",
      "maize",
      "rice",
      "wheat",
      "soybean",
      "cotton",
      "rapeseed",
      "vegetables",
      "orchard",
      "greenhouse",
      "fallow",
      "water",
      "built_or_bare"
    ],
    "metrics": {
      "readiness": "demo_contract_only",
      "validated_accuracy": null
    },
    "trained_region": "demo contract; no production crop checkpoint configured",
    "supported_sensors": ["multiband crop composite", "future Sentinel-2/HLS composite"],
    "license": "research_demo",
    "status": "demo_only",
    "example_inputs": ["examples/prithvi_crop_cached_demo.json"],
    "package_profile": {
      "package_type": "arcgis_style_pretrained_imagery_model",
      "family": "prithvi_crop_classification",
      "runtime_modes": ["cached_demo"],
      "input_profile": {
        "raster_profile": "multiband_crop_composite",
        "requires_georeferencing": true,
        "requires_crop_season_composite": true,
        "notes": "Demo contract only; real Prithvi crop-head inference is a later phase."
      },
      "output_profile": {
        "primary_output": "categorical crop raster",
        "artifacts": ["preview_png", "crop_polygons_geojson", "area_summary_csv"],
        "class_count": 13
      },
      "applicability": {
        "readiness": "demo_contract_only",
        "region": "not production validated",
        "limitations": [
          "No crop checkpoint is wired in this phase.",
          "No ArcGIS .dlpk compatibility is claimed.",
          "Cached demo results are deterministic product-contract artifacts."
        ]
      },
      "model_card": {
        "summary": "ArcGIS-style Prithvi crop classification package contract for Model Hub demos.",
        "usage": "Run cached_demo to inspect result schema, crop class summaries, and GIS artifact manifests.",
        "next_step": "Attach a validated Prithvi crop head and multiband raster preprocessing pipeline."
      }
    }
  },
```

- [ ] **Step 5: Run registry tests**

Run:

```bash
python -m pytest tests/test_model_hub_registry.py -q
```

Expected: all registry tests pass.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add ae_backend/app/services/model_hub_registry.py ae_backend/app/data/model_hub_models.json tests/test_model_hub_registry.py
git commit -m "feat: add prithvi crop model package registry"
```

## Task 2: Crop Demo Runtime

**Files:**
- Create: `ae_backend/app/services/model_hub_crop.py`
- Create: `tests/test_model_hub_crop.py`

- [ ] **Step 1: Write failing crop runtime tests**

Create `tests/test_model_hub_crop.py`:

```python
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_summarize_cached_crop_demo_returns_crop_result_and_artifacts(tmp_path: Path):
    from app.services.model_hub_crop import summarize_cached_crop_demo

    crop_dir = tmp_path / "prithvi_crop_demo"
    crop_dir.mkdir()
    (crop_dir / "crop_preview.png").write_bytes(b"png")
    (crop_dir / "crop_polygons.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    (crop_dir / "crop_summary.csv").write_text(
        "class,pixels,fraction\nmaize,6400,0.64\n",
        encoding="utf-8",
    )

    result = summarize_cached_crop_demo(options={"crop_dir": str(crop_dir)})

    assert result["result"]["task"] == "crop_classification"
    assert result["result"]["model_id"] == "prithvi_crop_classification_arcgis_style"
    assert result["result"]["summary"]["dominant_class"] == "maize"
    assert result["result"]["summary"]["class_pixel_counts"]["maize"] == 6400
    assert result["result"]["summary"]["class_area_fraction"]["maize"] == 0.64
    assert result["result"]["model_package"]["package_type"] == "arcgis_style_pretrained_imagery_model"
    assert {artifact["kind"] for artifact in result["artifacts"]} == {"png", "geojson", "csv"}
    assert all(Path(artifact["path"]).name in {"crop_preview.png", "crop_polygons.geojson", "crop_summary.csv"} for artifact in result["artifacts"])


def test_summarize_cached_crop_demo_returns_planned_artifact_paths_without_files(tmp_path: Path):
    from app.services.model_hub_crop import summarize_cached_crop_demo

    crop_dir = tmp_path / "empty_crop_demo"

    result = summarize_cached_crop_demo(options={"crop_dir": str(crop_dir)})

    assert result["result"]["summary"]["dominant_class"] == "maize"
    assert {artifact["kind"] for artifact in result["artifacts"]} == {"png", "geojson", "csv"}
    assert any("planned artifact paths" in log for log in result["logs"])
```

- [ ] **Step 2: Run crop runtime tests to verify failure**

Run:

```bash
python -m pytest tests/test_model_hub_crop.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'app.services.model_hub_crop'`.

- [ ] **Step 3: Implement crop runtime service**

Create `ae_backend/app/services/model_hub_crop.py`:

```python
from __future__ import annotations

from pathlib import Path

from app.core.config import PROJECT_ROOT


CROP_MODEL_ID = "prithvi_crop_classification_arcgis_style"
CROP_CLASSES = [
    "background",
    "maize",
    "rice",
    "wheat",
    "soybean",
    "cotton",
    "rapeseed",
    "vegetables",
    "orchard",
    "greenhouse",
    "fallow",
    "water",
    "built_or_bare",
]
DEMO_PIXEL_COUNTS = {
    "background": 1200,
    "maize": 6400,
    "rice": 1800,
    "wheat": 900,
    "soybean": 700,
    "cotton": 300,
    "rapeseed": 260,
    "vegetables": 420,
    "orchard": 360,
    "greenhouse": 180,
    "fallow": 500,
    "water": 240,
    "built_or_bare": 740,
}
ARTIFACT_FILES = [
    ("png", "crop_preview.png"),
    ("geojson", "crop_polygons.geojson"),
    ("csv", "crop_summary.csv"),
]


def default_crop_demo_dir() -> Path:
    return Path(PROJECT_ROOT) / "results" / "model_hub" / "prithvi_crop_demo"


def _area_fractions(counts: dict[str, int]) -> dict[str, float]:
    total = max(sum(counts.values()), 1)
    return {
        class_name: round(count / total, 6)
        for class_name, count in counts.items()
    }


def _artifact_manifest(crop_dir: Path) -> tuple[list[dict], bool]:
    artifacts = [
        {"kind": kind, "path": str(crop_dir / filename)}
        for kind, filename in ARTIFACT_FILES
    ]
    all_exist = all(Path(artifact["path"]).exists() for artifact in artifacts)
    return artifacts, all_exist


def summarize_cached_crop_demo(*, options: dict) -> dict:
    crop_dir = Path(options.get("crop_dir") or default_crop_demo_dir())
    counts = dict(DEMO_PIXEL_COUNTS)
    fractions = _area_fractions(counts)
    dominant_class = max(counts, key=counts.get)
    artifacts, artifacts_exist = _artifact_manifest(crop_dir)
    artifact_log = "loaded cached crop demo artifacts" if artifacts_exist else "returned planned artifact paths"

    return {
        "result": {
            "task": "crop_classification",
            "model_id": CROP_MODEL_ID,
            "summary": {
                "class_pixel_counts": counts,
                "class_area_fraction": fractions,
                "dominant_class": dominant_class,
                "method": "cached ArcGIS-style Prithvi crop package demo",
            },
            "model_package": {
                "package_type": "arcgis_style_pretrained_imagery_model",
                "family": "prithvi_crop_classification",
                "runtime_mode": "cached_demo",
                "class_schema": list(CROP_CLASSES),
            },
        },
        "artifacts": artifacts,
        "logs": [f"{artifact_log} from {crop_dir}"],
    }
```

- [ ] **Step 4: Run crop runtime tests**

Run:

```bash
python -m pytest tests/test_model_hub_crop.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add ae_backend/app/services/model_hub_crop.py tests/test_model_hub_crop.py
git commit -m "feat: add prithvi crop model hub demo runtime"
```

## Task 3: API Detail And Job Dispatch

**Files:**
- Modify: `ae_backend/app/services/model_hub_runtime.py`
- Modify: `ae_backend/app/api/model_hub.py`
- Modify: `tests/test_model_hub_api.py`

- [ ] **Step 1: Write failing API tests**

Append these tests to `tests/test_model_hub_api.py`:

```python
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
    assert body["package_profile"]["input_profile"]["raster_profile"] == "multiband_crop_composite"
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
        "class,pixels,fraction\nmaize,6400,0.64\n",
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
    assert body["result"]["summary"]["dominant_class"] == "maize"
    assert {artifact["kind"] for artifact in body["artifacts"]} == {"png", "geojson", "csv"}
```

- [ ] **Step 2: Run API tests to verify failure**

Run:

```bash
python -m pytest tests/test_model_hub_api.py::test_model_hub_returns_prithvi_crop_model_details tests/test_model_hub_api.py::test_model_hub_runs_prithvi_crop_cached_demo_job -q
```

Expected: the detail test may pass after Task 1, but the job test fails because `create_job()` does not execute the crop model synchronously and `run_model_hub_job()` has no crop dispatch branch.

- [ ] **Step 3: Add runtime dispatch**

Modify `ae_backend/app/services/model_hub_runtime.py`.

Add this branch before the final `raise ModelHubRuntimeError`:

```python
    if model_id == "prithvi_crop_classification_arcgis_style" and input_mode == "cached_demo":
        from app.services.model_hub_crop import summarize_cached_crop_demo

        return summarize_cached_crop_demo(options=options)
```

- [ ] **Step 4: Add synchronous API execution condition**

Modify the `should_execute_now` expression in `ae_backend/app/api/model_hub.py` to:

```python
    should_execute_now = (
        (
            request.model_id == "lulc_6class_prithvi_houlsby"
            and request.input_mode == "demo_patch"
        )
        or (
            request.model_id == "semantic_change_prithvi"
            and request.input_mode == "cached_demo"
        )
        or (
            request.model_id == "prithvi_crop_classification_arcgis_style"
            and request.input_mode == "cached_demo"
        )
    )
```

- [ ] **Step 5: Run API tests**

Run:

```bash
python -m pytest tests/test_model_hub_api.py tests/test_model_hub_crop.py -q
```

Expected: all API and crop runtime tests pass.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add ae_backend/app/services/model_hub_runtime.py ae_backend/app/api/model_hub.py tests/test_model_hub_api.py
git commit -m "feat: run prithvi crop model hub demo jobs"
```

## Task 4: Frontend Runtime-Mode Contract

**Files:**
- Modify: `ae_frontend/index.html`
- Modify: `tests/test_model_hub_frontend_entry.py`

- [ ] **Step 1: Write failing frontend contract test**

Append this test to `tests/test_model_hub_frontend_entry.py`:

```python
def test_frontend_uses_model_hub_runtime_modes_for_demo_jobs():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "getModelHubDemoInputMode" in html
    assert "package_profile?.runtime_modes" in html
    assert "default_demo_input_mode" in html
    assert "prithvi_crop_classification_arcgis_style" not in html
```

- [ ] **Step 2: Run frontend test to verify failure**

Run:

```bash
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: fail because the frontend still hard-codes runnable model ids and input mode logic.

- [ ] **Step 3: Replace hard-coded runnable logic**

In `ae_frontend/index.html`, replace:

```javascript
            const isModelHubRunnable = (model) => {
                return model && [
                    'lulc_6class_prithvi_houlsby',
                    'semantic_change_prithvi',
                ].includes(model.model_id);
            };
```

with:

```javascript
            const getModelHubDemoInputMode = (model) => {
                if (!model) return null;
                const runtimeModes = model.package_profile?.runtime_modes || [];
                if (runtimeModes.includes('cached_demo')) return 'cached_demo';
                if (model.input_spec?.default_demo_input_mode) return model.input_spec.default_demo_input_mode;
                if (model.model_id === 'lulc_6class_prithvi_houlsby') return 'demo_patch';
                return null;
            };

            const isModelHubRunnable = (model) => {
                return Boolean(getModelHubDemoInputMode(model));
            };
```

- [ ] **Step 4: Use metadata-selected input mode**

In `runModelHubDemo`, replace:

```javascript
                const inputMode = model.model_id === 'semantic_change_prithvi' ? 'cached_demo' : 'demo_patch';
```

with:

```javascript
                const inputMode = getModelHubDemoInputMode(model);
                if (!inputMode) {
                    setModelHubStatus('info', `${model.display_name} 浠嶅湪璁″垝涓€俙);
                    return;
                }
```

The existing early `if (!isModelHubRunnable(model))` block can stay. The duplicate guard is intentional because `runModelHubDemo()` is also a callable method, not only a button handler.

- [ ] **Step 5: Return helper from Vue setup**

In the `return` object near the existing Model Hub entries, replace:

```javascript
                runningModelHubJob, fetchModelHubModels, isModelHubRunnable, runModelHubDemo,
```

with:

```javascript
                runningModelHubJob, fetchModelHubModels, getModelHubDemoInputMode, isModelHubRunnable, runModelHubDemo,
```

- [ ] **Step 6: Run frontend contract test**

Run:

```bash
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: frontend contract tests pass.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add ae_frontend/index.html tests/test_model_hub_frontend_entry.py
git commit -m "feat: make model hub demo mode metadata driven"
```

## Task 5: End-To-End Verification

**Files:**
- Modify only files already touched if verification exposes an issue.

- [ ] **Step 1: Run focused Model Hub test suite**

Run:

```bash
python -m pytest tests/test_model_hub_registry.py tests/test_model_hub_crop.py tests/test_model_hub_jobs.py tests/test_model_hub_api.py tests/test_model_hub_change.py tests/test_raster_pipeline.py tests/test_model_hub_frontend_entry.py tests/test_inference_api.py tests/test_lulc_registry.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run Paper 12 regression tests**

Run:

```bash
python -m pytest tests/test_paper12_public_dataset_results.py tests/test_paper12_colab_notebooks.py -q
```

Expected: both Paper 12 regression files pass.

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Inspect repository state**

Run:

```bash
git status --short --branch
git log --oneline -8
```

Expected: the branch contains the Phase 2 design commit, the implementation commits from Tasks 1-4, and no uncommitted changes unless Step 5 produced a verification fix.

- [ ] **Step 5: Commit verification fixes if needed**

If verification exposed a small fix, commit only touched Phase 2 files:

```bash
git add ae_backend/app/services/model_hub_registry.py ae_backend/app/data/model_hub_models.json ae_backend/app/services/model_hub_crop.py ae_backend/app/services/model_hub_runtime.py ae_backend/app/api/model_hub.py ae_frontend/index.html tests/test_model_hub_registry.py tests/test_model_hub_crop.py tests/test_model_hub_api.py tests/test_model_hub_frontend_entry.py
git commit -m "test: verify prithvi crop model hub phase 2"
```

If there are no uncommitted changes, record the passing commands in the final implementation report.

## Spec Coverage Map

- Registry optional package metadata: Task 1.
- Crop model package entry: Task 1.
- Deterministic cached crop demo runtime: Task 2.
- Existing Model Hub API detail and job contracts: Task 3.
- Metadata-driven frontend run behavior: Task 4.
- Regression and whitespace verification: Task 5.

## Execution Notes

- Keep `prithvi_crop_classification_arcgis_style` at `status=demo_only`.
- Do not add real Prithvi weights, checkpoints, downloads, or GPU paths.
- Do not make `.dlpk` compatibility claims in code or UI text.
- Preserve the existing LULC and semantic-change behavior.
- Follow TDD strictly: write the test, run it to observe the expected failure, implement the minimal code, then rerun.
