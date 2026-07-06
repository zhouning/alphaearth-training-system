# Paper 12 Model-Hub UI Integration Design

## Objective

Integrate the current Paper 12 Prithvi/PEFT model work into AlphaEarth System as a user-operable model-center workflow. The feature must make the available capabilities usable from the existing UI while clearly separating production-ready models, contract demos, and planned work.

The immediate goal is a transparent productized loop:

- Show Paper 12 benchmark evidence inside the system.
- Show which Prithvi-derived capabilities are ready, demo-only, or planned.
- Let users run the currently supported model-hub jobs from the UI.
- Preserve clear limitations for ArcGIS-style crop classification and other Prithvi capabilities that do not yet have real local checkpoints.

## Current Context

AlphaEarth System already has:

- A static Vue 3 single-page frontend in `ae_frontend/index.html`.
- A FastAPI model hub mounted at `/api/ae/model-hub`.
- A model registry in `ae_backend/app/data/model_hub_models.json`.
- Runtime jobs for:
  - `lulc_6class_prithvi_houlsby` with `demo_patch`.
  - `semantic_change_prithvi` with `cached_demo`.
  - `prithvi_crop_classification_arcgis_style` with `cached_demo` and `upload_raster_demo`.
- Paper 12 result files in `paper12_results/`, including EuroSAT, LandCover.ai, Linhe LULC, LoveDA, and capacity-sweep outputs.

The current crop raster runtime validates and processes an 18-band GeoTIFF, but it is explicitly a deterministic contract demo. It does not load a real Prithvi crop checkpoint and must not be presented as accuracy-validated crop inference.

## User Experience

Enhance the existing `模型中心` tab rather than creating a new application.

The page should contain four clear areas:

1. Paper 12 capability summary
   - Counts of ready, demo-only, and planned models.
   - Key benchmark evidence from Paper 12.
   - A short readiness note explaining that the page exposes research/demo capabilities, not full ArcGIS replacement.

2. Model cards
   - Model name, task type, backbone, adapter, status, supported sensors, and trained region.
   - Metric chips when available.
   - Input requirements and output type.
   - Limitation text from registry metadata when present.

3. Job controls
   - A general demo button when the model has a runnable demo mode.
   - A crop raster form only for models whose runtime modes include `upload_raster_demo`.
   - The raster form should explain accepted input paths and 18-band requirements.
   - Buttons must be disabled while a job is running.

4. Result summary
   - Job status and error display.
   - Human-readable result summary before raw JSON.
   - Artifact list with paths.
   - Runtime logs.
   - Raw JSON kept as a detail/debug view.

The interface should remain quiet and operational, matching the existing dashboard style. It should use compact status chips and dense cards rather than a landing-page layout.

## API Design

Add a read-only endpoint:

`GET /api/ae/model-hub/paper12-summary`

Response shape:

```json
{
  "paper": "paper12",
  "readiness_counts": {
    "ready": 1,
    "demo_only": 2,
    "planned": 3
  },
  "benchmarks": [
    {
      "id": "eurosat_channel_bridge",
      "label": "EuroSAT channel bridge",
      "metric": "overall_accuracy",
      "best_method": "learned_bridge_houlsby",
      "best_value": 0.9095679012345679,
      "source": "paper12_results/eurosat_channel_bridge_summary.json"
    }
  ],
  "capabilities": [
    {
      "model_id": "prithvi_crop_classification_arcgis_style",
      "readiness": "demo_only",
      "arcgis_replacement_status": "not_yet",
      "reason": "No validated crop checkpoint is configured; current runtime is a deterministic contract demo.",
      "next_step": "Attach a validated Prithvi crop head and HLS preprocessing pipeline."
    }
  ]
}
```

The endpoint should compute values from the committed model registry and local Paper 12 result files. If a result file is missing, the endpoint should still return successfully with a missing-data note for that benchmark instead of failing the whole page.

## Backend Components

Add a small service module, for example:

`ae_backend/app/services/paper12_summary.py`

Responsibilities:

- Load `paper12_results/*summary*.json` and `landcoverai_segmentation.json`.
- Summarize only the metrics needed by the UI.
- Compute model readiness counts from `ModelHubRegistry`.
- Emit explicit capability readiness and ArcGIS-replacement status from registry metadata.
- Avoid importing heavy ML libraries.

Extend `ae_backend/app/api/model_hub.py` with the new route. The route should reuse `get_model_registry()` and the new summary service.

## Frontend Components

Modify only `ae_frontend/index.html`.

Add state and methods:

- `paper12Summary`
- `loadingPaper12Summary`
- `fetchPaper12Summary`
- `statusClass(status)`
- `formatMetric(value)`
- `summarizeModelHubJob(job)`

Update the `modelHub` tab markup:

- Add the Paper 12 summary band above model cards.
- Improve each model card with status chips and limitations.
- Keep existing `runModelHubDemo` and `runModelHubRasterDemo` flows.
- Replace raw-JSON-only result display with summary, artifacts, logs, and raw JSON.

The frontend should continue to pass existing tests that assert model-hub APIs and crop raster controls are exposed.

## Testing

Use test-first implementation.

Backend tests:

- `GET /api/ae/model-hub/paper12-summary` returns 200.
- Readiness counts include committed registry statuses.
- EuroSAT summary reports `learned_bridge_houlsby` as the best channel-bridge method.
- LandCover.ai summary reports Houlsby as the best segmentation method.
- Crop capability is marked `demo_only` and `arcgis_replacement_status: not_yet`.
- Missing optional result files produce missing-data notes rather than endpoint failure.

Frontend tests:

- `ae_frontend/index.html` contains `paper12Summary`, `fetchPaper12Summary`, and `/api/ae/model-hub/paper12-summary`.
- The model hub tab exposes Paper 12 capability summary text.
- The page exposes model status chips, limitation text, artifact list, logs, and raw JSON details.
- Existing crop raster controls remain metadata-driven.

Focused verification:

- `python -m pytest tests/test_model_hub_api.py tests/test_model_hub_registry.py tests/test_model_hub_frontend_entry.py`

If dependencies are available, also run:

- `python -m pytest tests/test_paper12_public_dataset_results.py`
- `git diff --check`

## Non-Goals

- Do not claim full replacement of ArcGIS Prithvi pretrained models.
- Do not add real crop, flood, burn-scar, weather, or Prithvi EO 2.0 checkpoints in this pass.
- Do not build ArcGIS `.dlpk` compatibility.
- Do not introduce a new frontend framework or split the static page into a build system.
- Do not change Paper 12 manuscript text or result files.
- Do not change model training behavior.

## Acceptance Criteria

- A user opening the existing AlphaEarth System UI can enter `模型中心`, see Paper 12 capability evidence, identify model readiness, and run all currently supported model-hub demos.
- The UI clearly warns when a capability is only a contract demo.
- Backend summary data is served from committed local files and registry metadata.
- Existing model-hub functionality remains compatible.
- Focused backend and frontend contract tests pass.
