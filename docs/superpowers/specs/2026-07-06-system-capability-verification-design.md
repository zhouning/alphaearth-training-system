# System Capability Verification Design

Date: 2026-07-06
Worktree: `D:\adk\AlphaEarth-System`
Branch: `master`

## Objective

Extend the system-first capability workbench from "what the system claims it
can do" to "what the system can verify about those claims". This slice adds a
deterministic self-check layer for AlphaEarth System capabilities. The output
should help users understand why a model is runnable, why a demo is not a
validated replacement, which evidence files exist, and what action is needed
next.

Paper 12 remains downstream. The manuscript can later cite the system's
verified outputs, but this pass does not edit manuscript text or invent new
experimental results.

## Current Context

The previous system-first slice added:

- `GET /api/ae/system/capabilities`.
- `ae_backend/app/services/system_capabilities.py`.
- `ae_backend/app/api/system.py`.
- Model Hub UI integration in `ae_frontend/index.html`.
- Contract tests for the backend endpoint and frontend workbench.

The current endpoint reports capability status from registry metadata, runtime
modes, checkpoint configuration, and evidence-source availability. It does not
yet produce a first-class audit trail of checks, warnings, failures, and
remediation actions.

## User Experience

Enhance the existing System Capability Workbench in the Model Hub tab. Do not
create a new application or landing page.

Add a compact verification area with:

- Overall verification status: `pass`, `warning`, or `fail`.
- Count chips for `pass`, `warning`, `fail`, and `not_applicable`.
- Per-capability verification status on each model card.
- A small list of blocking issues and recommended next actions.
- A raw JSON detail block for debugging.

The verification area should make these distinctions clear:

- A configured runnable LULC capability can pass operational checks while still
  carrying applicability limitations.
- A deterministic crop contract demo can pass contract checks while failing or
  warning on validated replacement readiness.
- Planned models should not be treated as failures simply because checkpoints
  are absent; their checks should be `not_applicable` or `warning` with clear
  next actions.
- Missing optional Paper 12 evidence files should produce warnings, not crash
  the page.

## API Design

Add a read-only endpoint:

`GET /api/ae/system/verification`

Response shape:

```json
{
  "system": "AlphaEarth System",
  "generated_at": "2026-07-06T00:00:00Z",
  "overall_status": "warning",
  "summary": {
    "pass": 8,
    "warning": 3,
    "fail": 0,
    "not_applicable": 5
  },
  "capabilities": [
    {
      "id": "lulc_6class_prithvi_houlsby",
      "overall_status": "pass",
      "checks": [
        "lulc_6class_prithvi_houlsby:registry_status",
        "lulc_6class_prithvi_houlsby:checkpoint_configured"
      ],
      "blocking_issues": [],
      "next_actions": []
    },
    {
      "id": "prithvi_crop_classification_arcgis_style",
      "overall_status": "warning",
      "checks": [
        "prithvi_crop_classification_arcgis_style:contract_demo_declared",
        "prithvi_crop_classification_arcgis_style:arcgis_replacement_guard"
      ],
      "blocking_issues": [],
      "next_actions": [
        "Attach a validated Prithvi crop checkpoint before claiming replacement readiness."
      ]
    }
  ],
  "checks": [
    {
      "id": "prithvi_crop_classification_arcgis_style:arcgis_replacement_guard",
      "capability_id": "prithvi_crop_classification_arcgis_style",
      "category": "replacement_boundary",
      "status": "pass",
      "severity": "info",
      "title": "ArcGIS replacement guard is conservative",
      "detail": "The crop capability is marked as demo-only and not a validated ArcGIS replacement.",
      "evidence_refs": ["model_hub_registry"],
      "remediation": null
    }
  ],
  "notes": [
    "Verification is deterministic and does not load model weights.",
    "A pass means the declared system contract is internally consistent, not that global production accuracy is proven."
  ]
}
```

Status semantics:

- `pass`: The declared system contract is internally consistent.
- `warning`: The system can continue, but the user should see a limitation or
  missing optional evidence.
- `fail`: A declared runnable capability is internally inconsistent or missing
  a required configuration.
- `not_applicable`: The check does not apply to a planned or unsupported
  capability.

## Backend Components

Add a focused verification service:

`ae_backend/app/services/system_verification.py`

Responsibilities:

- Reuse `ModelHubRegistry` entries.
- Optionally reuse `build_system_capabilities()` output to avoid duplicating
  readiness logic.
- Build deterministic check records.
- Summarize check status per capability and globally.
- Avoid PyTorch, rasterio, model loading, and network access.
- Treat optional evidence files as warnings.
- Keep ArcGIS replacement checks conservative.

Add route to the existing system API:

- Modify `ae_backend/app/api/system.py`.
- Add `GET /verification`.
- Reuse `get_model_registry()` from the existing system route pattern.

## Verification Checks

Minimum checks for this slice:

1. Registry status check
   - Every registered model must have a known status.
   - Unknown status is `fail`.

2. Runtime mode check
   - Ready or demo-only entries should expose at least one runtime mode when
     they are expected to be executable.
   - Planned entries receive `not_applicable`.

3. Checkpoint configuration check
   - Ready entries should have a checkpoint path configured.
   - Demo-only contract entries may pass with no checkpoint only when their
     workflow level is explicitly `contract_demo` or `demo`.
   - Planned entries receive `not_applicable`.

4. Evidence source check
   - Existing evidence files are `pass`.
   - Missing optional evidence files are `warning`.

5. Replacement boundary check
   - ArcGIS-style crop capability must remain `not_ready` unless a validated
     checkpoint-backed workflow is configured.
   - A conservative guard is `pass`; an overclaim is `fail`.

## Frontend Components

Modify only `ae_frontend/index.html` in the implementation pass.

Add state and methods:

- `systemVerification`
- `loadingSystemVerification`
- `systemVerificationError`
- `fetchSystemVerification`
- `verificationStatusClass(status)`
- `verificationForCapability(model)`

Update the Model Hub workbench:

- Add a verification summary band below or inside the existing system readiness
  overview.
- Add per-card verification chips.
- Show a compact list of next actions for warning/fail capabilities.
- Add a collapsible raw JSON detail for verification data.
- Keep current model-hub job controls unchanged.

## Testing

Use test-first implementation.

Backend tests:

- `GET /api/ae/system/verification` returns 200.
- Response includes `overall_status`, `summary`, `capabilities`, `checks`, and
  `notes`.
- LULC capability has an operational pass or no blocking issue.
- Crop capability includes a replacement-boundary check that prevents ArcGIS
  overclaiming.
- Planned capabilities are not counted as failures solely because checkpoints
  are missing.
- Missing optional evidence files become warnings.

Frontend tests:

- `ae_frontend/index.html` references `/api/ae/system/verification`.
- Frontend state includes `systemVerification` and
  `fetchSystemVerification`.
- Model Hub UI exposes verification status, next actions, and raw JSON detail
  hooks.
- Existing system capability and model-hub job controls remain present.

Focused verification:

- `python -m pytest tests/test_model_hub_api.py tests/test_model_hub_frontend_entry.py`
- `python -m pytest tests/test_model_hub_registry.py tests/test_inference_api.py tests/test_inference_service.py`
- `git diff --check`

## Non-Goals

- Do not run heavy model inference as part of verification.
- Do not train new models.
- Do not validate global production accuracy.
- Do not claim full ArcGIS replacement readiness.
- Do not edit Paper 12 manuscript files.
- Do not add a frontend build system.

## Acceptance Criteria

- The backend exposes deterministic verification data at
  `/api/ae/system/verification`.
- The UI shows system verification status and next actions in the existing
  Model Hub workbench.
- Ready, demo-only, planned, and missing-evidence cases are distinguishable.
- Crop classification remains guarded against ArcGIS replacement overclaiming.
- Focused tests pass.

## Implementation Boundary

This slice turns capability status into a verifiable system contract. Later
work can add clickable artifact previews, richer LULC evaluation loops, or real
checkpoint-backed model expansion, but those should build on this verification
layer.
