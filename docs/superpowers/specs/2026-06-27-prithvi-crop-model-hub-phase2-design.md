# Prithvi Crop Model Hub Phase 2 Design

Date: 2026-06-27
Worktree: `D:\tmp\alphaearth-paper12-results-20260619`
Branch: `paper12-results-colab-20260619`

## Goal

Phase 2 adds an ArcGIS Prithvi Crop Classification-style model package to the
existing Paper 12 Model Hub. A user should be able to discover the crop model,
inspect its model-card metadata, start a deterministic demo job, and receive a
GIS-shaped result manifest without writing PyTorch code or depending on ArcGIS
Pro.

This phase turns the Phase 1 Model Hub from a generic Prithvi demo catalog into
a closer commercial analogue of ArcGIS pretrained imagery models: model package
metadata, explicit input requirements, declared applicability, job execution,
and downloadable result artifacts.

## Product Target

The new model entry is:

- `model_id`: `prithvi_crop_classification_arcgis_style`
- `display_name`: `Prithvi Crop Classification Package`
- `task_type`: `crop_classification`
- `status`: `demo_only`
- `backbone`: `Prithvi-100M`
- `adapter`: `crop classification head contract`

The package is not a claim that this repository already reproduces Esri's
production model. It is a product-contract implementation that mirrors the
experience users expect from ArcGIS AI models:

1. A clear model card.
2. A declared input raster profile.
3. A categorical crop output schema.
4. Applicability and limitation notes.
5. A runnable demo job through the same Model Hub job API.
6. Result artifacts that downstream GIS tools can consume.

## Existing Assets To Reuse

- `ae_backend/app/services/model_hub_registry.py`: validates and serves model
  metadata.
- `ae_backend/app/data/model_hub_models.json`: Phase 1 model registry.
- `ae_backend/app/api/model_hub.py`: existing model and job endpoints.
- `ae_backend/app/services/model_hub_runtime.py`: job dispatch.
- `ae_backend/app/services/model_hub_jobs.py`: in-memory job lifecycle.
- `ae_backend/app/services/raster_pipeline.py`: class-area summary utilities.
- `tests/test_model_hub_registry.py`, `tests/test_model_hub_api.py`,
  `tests/test_model_hub_change.py`: patterns for registry/API/runtime tests.
- `docs/superpowers/specs/2026-06-26-commercial-model-hub-phase1-design.md`:
  approved Phase 1 boundary.

## Non-Goals

- Do not download Prithvi weights.
- Do not add GPU execution or real crop-head inference.
- Do not introduce an ArcGIS Pro or ArcPy runtime dependency.
- Do not generate a real Esri `.dlpk` package.
- Do not claim global or production crop-classification accuracy.
- Do not make building, road, or water models runnable in this phase.

## Model Package Contract

The registry entry should keep all existing required fields and add optional
package metadata under a `package_profile` object. The registry must preserve
unknown optional fields in public payloads so future model packages can add
metadata without changing the registry schema each time.

Required package profile fields for this model:

- `package_type`: `arcgis_style_pretrained_imagery_model`
- `family`: `prithvi_crop_classification`
- `runtime_modes`: `["cached_demo"]`
- `input_profile`: describes a multiband crop-composite raster contract.
- `output_profile`: describes a categorical crop raster plus summary artifacts.
- `applicability`: states region, sensor, and validation limitations.
- `model_card`: short usage, license, and readiness text for the frontend.

The demo class schema is local to this repository. It should be explicit and
stable:

1. `background`
2. `maize`
3. `rice`
4. `wheat`
5. `soybean`
6. `cotton`
7. `rapeseed`
8. `vegetables`
9. `orchard`
10. `greenhouse`
11. `fallow`
12. `water`
13. `built_or_bare`

The registry status remains `demo_only` because no validated crop model
checkpoint is wired in yet.

## Runtime Design

Add `ae_backend/app/services/model_hub_crop.py` with one public function:

```python
def summarize_cached_crop_demo(*, options: dict) -> dict:
    """Return a deterministic Model Hub crop-classification demo result."""
```

The function returns the same shape as the Phase 1 runtimes:

```python
{
    "result": {
        "task": "crop_classification",
        "model_id": "prithvi_crop_classification_arcgis_style",
        "summary": {
            "class_pixel_counts": {"maize": 6400, "rice": 1800, "background": 1200},
            "class_area_fraction": {"maize": 0.64, "rice": 0.18, "background": 0.12},
            "dominant_class": "maize",
            "method": "cached ArcGIS-style Prithvi crop package demo"
        },
        "model_package": {"package_type": "arcgis_style_pretrained_imagery_model"}
    },
    "artifacts": [
        {"kind": "png", "path": "results/model_hub/prithvi_crop_demo/crop_preview.png"},
        {"kind": "geojson", "path": "results/model_hub/prithvi_crop_demo/crop_polygons.geojson"},
        {"kind": "csv", "path": "results/model_hub/prithvi_crop_demo/crop_summary.csv"}
    ],
    "logs": ["loaded cached Prithvi crop classification demo"]
}
```

The default demo can be deterministic and file-light. If no demo artifact files
exist, the service may return planned paths under `results/model_hub/prithvi_crop_demo`
and an inline summary. Tests should use temporary files to verify artifact
discovery without requiring large binary assets in git.

## API Design

Reuse existing endpoints:

- `GET /api/ae/model-hub/models`
- `GET /api/ae/model-hub/models/{model_id}`
- `POST /api/ae/model-hub/jobs`
- `GET /api/ae/model-hub/jobs/{job_id}`

The new runnable request is:

```json
{
  "model_id": "prithvi_crop_classification_arcgis_style",
  "input_mode": "cached_demo",
  "options": {
    "output_formats": ["png", "geojson", "csv"]
  }
}
```

The job should complete synchronously like the current LULC and change demo
jobs. Unsupported input modes should remain pending or fail through the existing
runtime error path; this phase only makes `cached_demo` runnable.

## Frontend Boundary

The existing Model Hub page should list the new model automatically from the
registry. No new tab is required in this phase.

If the current demo-run button already sends `cached_demo` for non-LULC models,
the crop model can use that behavior unchanged. If the frontend has hard-coded
logic that only treats change detection as `cached_demo`, update the logic so
models declare their preferred demo input mode in `package_profile.runtime_modes`
or `input_spec.default_demo_input_mode`.

## Testing Strategy

Write tests before implementation:

1. Registry test: committed registry loads six models and preserves the crop
   model's `package_profile`.
2. API detail test: the crop model detail includes `task_type`,
   `package_profile.input_profile`, `package_profile.output_profile`, and
   `status=demo_only`.
3. Crop runtime unit test: `summarize_cached_crop_demo` returns
   `task=crop_classification`, a dominant class, class-area summary, and artifact
   kinds `png`, `geojson`, and `csv`.
4. API job test: posting the crop cached demo job returns `status=succeeded` and
   the crop classification result.
5. Regression tests: existing Phase 1 Model Hub tests and Paper 12 notebook/result
   tests still pass.

## Acceptance Criteria

Phase 2 is complete when:

- `GET /api/ae/model-hub/models` includes
  `prithvi_crop_classification_arcgis_style`.
- The crop model detail exposes ArcGIS-style model-package metadata and
  applicability limitations.
- The crop model can be run through `POST /api/ae/model-hub/jobs` with
  `input_mode=cached_demo`.
- The job result includes class pixel counts, class area fractions, dominant
  class, model package metadata, and artifact manifests.
- Existing LULC and semantic change jobs keep their behavior.
- Focused tests pass.
- `git diff --check` is clean.

## Implementation Milestones

### Milestone 1: Registry Extensibility

- Preserve optional metadata fields in `ModelHubEntry`.
- Add the crop classification model entry to `model_hub_models.json`.
- Add tests proving optional package metadata is returned publicly.

### Milestone 2: Crop Demo Runtime

- Add `model_hub_crop.py`.
- Implement deterministic cached crop summary and artifact manifest discovery.
- Add unit tests for runtime behavior and missing/empty artifact directories.

### Milestone 3: Job API Dispatch

- Add a dispatch branch in `model_hub_runtime.py`.
- Update `model_hub.py` synchronous execution condition.
- Add API tests for successful crop demo jobs.

### Milestone 4: Frontend Contract

- Verify the new model appears through registry-driven frontend behavior.
- If needed, update demo-run mode selection to read registry metadata instead
  of hard-coding only LULC and change detection.

### Milestone 5: Verification

- Run focused Model Hub tests.
- Run Paper 12 regression tests.
- Run `git diff --check`.

## Risks And Mitigations

- Risk: product metadata overstates model readiness.
  Mitigation: keep `status=demo_only` and include explicit applicability notes.

- Risk: optional registry fields break existing tests.
  Mitigation: preserve existing fields and only add optional metadata passthrough.

- Risk: frontend cannot choose the right demo input mode.
  Mitigation: expose a default demo mode in model metadata and update the button
  only if the existing behavior is insufficient.

- Risk: users confuse the local package with Esri's production `.dlpk`.
  Mitigation: call it ArcGIS-style, not ArcGIS-compatible, and list `.dlpk`
  export as a later phase.

## Later Phases

- Add real Prithvi crop-head checkpoint loading and inference.
- Add real multiband raster validation and tiled crop-raster export.
- Add `.dlpk`-like package export or an Esri interoperability manifest.
- Add validated regional crop datasets and accuracy metrics.
- Add asynchronous job execution for larger rasters.

## References

- Existing Phase 1 spec:
  `docs/superpowers/specs/2026-06-26-commercial-model-hub-phase1-design.md`
- ArcGIS AI models overview:
  `https://doc.arcgis.com/en/pretrained-models/latest/get-started/intro.htm`
- ArcGIS Prithvi Crop Classification reference:
  `https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-prithvi-crop-classification.htm`
