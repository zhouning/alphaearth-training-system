# System Capability Workbench Design

Date: 2026-07-06
Worktree: `D:\adk\AlphaEarth-System`
Branch: `master`

## Objective

Prioritize AlphaEarth System as an operable remote-sensing product before any
further Paper 12 manuscript work. The system should expose what it can run,
what evidence supports each capability, and what still remains demo-only or
planned. Paper 12 becomes a downstream consumer of system outputs and measured
effects, not the primary driver of UI or backend design.

The immediate goal is a "System Capability Workbench" inside the existing web
application:

- Users can inspect the current model and workflow capabilities.
- Users can distinguish runnable, evaluable, demo-only, planned, and
  not-configured capabilities.
- Users can run the currently supported model-hub and LULC workflows from the
  UI.
- Users can see an evidence trail: registry metadata, checkpoints, runtime
  modes, benchmark artifacts, limitations, and last verification status.
- The system keeps conservative replacement language for ArcGIS/Prithvi-style
  capabilities. A local deterministic crop demo is not presented as a validated
  ArcGIS replacement.

## Current Context

AlphaEarth System already has:

- A static Vue 3 frontend in `ae_frontend/index.html`.
- A FastAPI backend mounted under `/api/ae`.
- Model Hub API routes in `ae_backend/app/api/model_hub.py`.
- LULC inference API routes in `ae_backend/app/api/inference.py`.
- A model registry in `ae_backend/app/data/model_hub_models.json`.
- Model Hub services for registry loading, in-memory jobs, runtime dispatch,
  crop demos, change demos, and Paper 12 summaries.
- A ready LULC entry, `lulc_6class_prithvi_houlsby`, backed by a configured
  Linhe checkpoint.
- Demo-only entries for semantic change and ArcGIS-style Prithvi crop
  classification.
- Planned entries for building, road/hard-surface, and water/flood tasks.

The existing `paper12-summary` endpoint and frontend panel are useful but framed
around the manuscript. This pass should refocus the product surface around the
system itself, then treat Paper 12 files as one evidence source among several.

## User Experience

Enhance the existing Model Hub / LULC UI rather than creating a new application
or a marketing page.

The workbench should contain five operational areas:

1. System readiness overview
   - Counts for `ready`, `evaluable`, `demo_only`, `planned`, and
     `not_configured` capabilities.
   - A concise operational statement such as "1 runnable model, 2 demo-only
     workflows, 3 planned workflows".
   - A warning when any ArcGIS-style capability is not validated for
     replacement use.

2. Capability cards
   - Capability name, model id, task type, status, runtime modes, input
     requirements, output type, supported sensors, trained region, and
     checkpoint presence.
   - Metric chips when committed evidence exists.
   - Limitation text and next-step text from registry or service metadata.
   - A visible status distinction between real runnable inference and
     deterministic contract demos.

3. Workflow controls
   - Keep current model-hub demo controls.
   - Keep current crop raster demo controls, but label them as contract/demo
     mode unless a validated crop checkpoint is added later.
   - Keep current LULC upload/evaluate controls available from the LULC tab.
   - Disable controls while a job is running and show backend error messages
     without hiding the capability card.

4. Evidence trail
   - Show registry source, checkpoint path or missing-checkpoint reason, runtime
     mode support, benchmark artifact paths, and available test/verification
     notes.
   - Include Paper 12 benchmark artifacts only as supporting evidence.
   - Prefer summarized values first, with raw JSON available as a detail view.

5. Result inspection
   - Human-readable job summary.
   - Artifact list.
   - Logs.
   - Raw JSON for debugging.
   - Clear labels for synthetic, cached, deterministic, or validated outputs.

The UI should stay compact and operational, matching the existing dashboard
style. Do not convert the page into a publication landing page.

## API Design

Add a system-first read-only endpoint:

`GET /api/ae/system/capabilities`

Proposed response shape:

```json
{
  "system": "AlphaEarth System",
  "generated_at": "2026-07-06T00:00:00Z",
  "readiness_counts": {
    "ready": 1,
    "evaluable": 1,
    "demo_only": 2,
    "planned": 3,
    "not_configured": 0
  },
  "summary": {
    "runnable_models": 1,
    "demo_workflows": 2,
    "planned_workflows": 3,
    "arcgis_replacement_ready": false
  },
  "capabilities": [
    {
      "id": "lulc_6class_prithvi_houlsby",
      "display_name": "Linhe 6-class LULC segmentation",
      "task_type": "semantic_segmentation",
      "readiness": "ready",
      "workflow_level": "runnable_and_evaluable",
      "runtime_modes": ["demo_patch"],
      "checkpoint": {
        "configured": true,
        "path": "linhe_lulc/houlsby__rgb_3band__seed123.pt"
      },
      "evidence": [
        {
          "kind": "metric",
          "label": "Linhe mIoU",
          "value": 0.2971,
          "source": "model_hub_registry"
        }
      ],
      "limitations": [],
      "next_steps": []
    },
    {
      "id": "prithvi_crop_classification_arcgis_style",
      "display_name": "Prithvi crop classification contract demo",
      "task_type": "crop_classification",
      "readiness": "demo_only",
      "workflow_level": "contract_demo",
      "runtime_modes": ["cached_demo", "upload_raster_demo"],
      "checkpoint": {
        "configured": false,
        "path": null
      },
      "arcgis_replacement": {
        "status": "not_ready",
        "reason": "No validated crop checkpoint or ArcGIS-compatible HLS preprocessing contract is configured."
      },
      "evidence": [],
      "limitations": [
        "Deterministic contract output; not validated crop inference."
      ],
      "next_steps": [
        "Attach a validated crop checkpoint and HLS preprocessing pipeline.",
        "Run independent crop validation before claiming replacement readiness."
      ]
    }
  ],
  "evidence_sources": [
    {
      "kind": "paper12_benchmark",
      "path": "paper12_results/eurosat_channel_bridge_summary.json",
      "available": true
    }
  ]
}
```

The endpoint should be lightweight and deterministic:

- No heavy ML imports.
- No model loading.
- No internet access.
- Missing optional evidence files should become notes, not endpoint failures.
- The endpoint should be safe to call repeatedly from the frontend.

The existing `/api/ae/model-hub/paper12-summary` endpoint may remain for
backward compatibility, but the UI should use the new system-first endpoint as
the primary source.

## Backend Components

Add a focused service, for example:

`ae_backend/app/services/system_capabilities.py`

Responsibilities:

- Load model registry entries through the existing registry service.
- Inspect runtime modes exposed by the model-hub runtime layer.
- Normalize capability readiness into product-facing values.
- Report checkpoint configured/missing status from registry metadata.
- Attach lightweight evidence from registry metrics and selected local result
  artifacts.
- Mark ArcGIS-style replacement readiness conservatively.
- Avoid imports from training, PyTorch, rasterio, or model-loading code.

Add an API module or extend an existing router:

- Prefer `ae_backend/app/api/system.py` if the current backend organization
  supports a new namespace cleanly.
- Otherwise extend `model_hub.py` with a clearly named route and mount path.

Update `ae_backend/app/main.py` only as needed to include the system router.

## Frontend Components

Modify only `ae_frontend/index.html` in this pass.

Add state and methods:

- `systemCapabilities`
- `loadingSystemCapabilities`
- `systemCapabilitiesError`
- `fetchSystemCapabilities`
- `capabilityStatusClass(status)`
- `capabilityWorkflowLabel(level)`
- `formatEvidenceValue(value)`

Update the Model Hub area:

- Replace the Paper 12-first summary band with a system readiness overview.
- Keep Paper 12 evidence visible only as an evidence source or detail.
- Add capability cards driven by `/api/ae/system/capabilities`.
- Keep existing model list and job controls compatible with current API
  responses.
- Improve result display so summary, artifacts, logs, and raw JSON are visible
  without requiring the user to parse raw objects first.

Update the LULC area only if needed to link it into the workbench:

- Keep existing upload/evaluation flows.
- Add a small readiness/evidence indicator from the system capability summary
  when the LULC capability is present.

## Data Flow

1. Frontend loads Model Hub tab.
2. Frontend requests `/api/ae/system/capabilities`.
3. Backend reads registry metadata and local evidence artifacts.
4. Backend returns product-facing capability status and evidence.
5. User runs an existing model-hub demo or LULC workflow.
6. Job/result panel shows output summary, artifacts, logs, and raw details.
7. Later manuscript work can cite these system-generated states and artifacts
   rather than driving the system design.

## Testing

Use test-first implementation.

Backend contract tests:

- `/api/ae/system/capabilities` returns 200.
- Response includes `system`, `readiness_counts`, `summary`,
  `capabilities`, and `evidence_sources`.
- LULC capability is reported as runnable when its registry checkpoint is
  configured.
- Crop classification is reported as `demo_only` or `contract_demo`, with
  `arcgis_replacement.status` equal to `not_ready`.
- Planned building, road, and water capabilities remain planned.
- Missing optional evidence files do not fail the endpoint.

Frontend tests:

- `ae_frontend/index.html` references `/api/ae/system/capabilities`.
- Frontend state includes `systemCapabilities` and
  `fetchSystemCapabilities`.
- The Model Hub UI contains system-first labels, not a Paper 12-first headline.
- The UI exposes readiness, evidence, limitations, artifacts, logs, and raw JSON
  detail hooks.
- Existing LULC and crop raster controls remain present.

Focused verification:

- `python -m pytest tests/test_model_hub_api.py tests/test_model_hub_registry.py tests/test_model_hub_frontend_entry.py`
- `python -m pytest tests/test_lulc_frontend_entry.py tests/test_inference_api.py tests/test_inference_service.py`
- `git diff --check`

## Non-Goals

- Do not edit Paper 12 manuscript text in this pass.
- Do not claim that AlphaEarth fully replaces ArcGIS Prithvi pretrained models.
- Do not add real crop, flood, building, or road checkpoints unless a validated
  local artifact already exists.
- Do not train new models in this pass.
- Do not add a frontend build system.
- Do not remove the existing Paper 12 summary endpoint unless all callers have
  been migrated and tests prove compatibility.

## Acceptance Criteria

- A user can open AlphaEarth System, enter the Model Hub area, and understand
  the system's current operational capabilities without reading the paper.
- The UI clearly separates validated runnable inference from deterministic
  contract demos and planned work.
- Current runnable workflows remain usable from the UI.
- Backend capability status is derived from registry/runtime/evidence files,
  not hard-coded publication claims.
- Paper 12 is represented only as supporting evidence.
- Focused backend and frontend contract tests pass.

## Implementation Boundary

This spec defines the first system-first implementation slice. Later work may
add richer raster inference, model training, independent validation datasets,
or manuscript updates, but those should be driven by system behavior and
measured outputs produced after this slice.
