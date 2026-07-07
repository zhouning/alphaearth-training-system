# System Evidence Drill-down Design

Date: 2026-07-07
Worktree: `D:\adk\AlphaEarth-System`
Branch: `master`

## Objective

Extend the system-first verification workbench from "what checks passed" to
"what evidence supports each check". This slice adds a deterministic evidence
drill-down layer that lets a user inspect local artifacts, understand missing
evidence, and trace verification claims back to concrete files or registry
records.

Paper 12 remains downstream. This work may expose artifacts that later support
the manuscript, but it does not edit manuscript text, regenerate experiment
results, or claim new model performance.

## Current Context

The previous system slices added:

- `GET /api/ae/system/capabilities`
- `GET /api/ae/system/verification`
- `ae_backend/app/services/system_capabilities.py`
- `ae_backend/app/services/system_verification.py`
- Model Hub UI integration for the System Capability Workbench.

The verification endpoint now returns check records with `evidence_refs`, but
the UI only shows check ids and raw JSON. Users cannot yet click a check and
see whether its referenced artifact exists, what kind of artifact it is, or a
small preview of inspectable JSON/CSV/text content.

## Recommended Approach

Implement a light drill-down panel inside the existing Model Hub System
Capability Workbench.

Do not create a new app, landing page, or large evidence browser in this slice.
The goal is to make the current verification output inspectable with minimal
new surface area:

- A compact evidence summary below the verification summary.
- Per-check evidence details under each model card.
- Small previews for safe text-like artifacts.
- File metadata for binary or large artifacts.
- Clear warnings for missing optional evidence.

## User Experience

Add an "Evidence drill-down" area in the existing Model Hub tab.

The UI should show:

- Evidence summary counts: available, missing, previewable, blocked.
- For each verification check, the check title/status and its evidence refs.
- For each evidence ref:
  - label
  - normalized relative path or registry source id
  - existence status
  - artifact kind
  - file size and modified time when available
  - short preview for supported small text artifacts
  - warning/remediation text when missing or blocked
- A refresh button that reloads evidence without rerunning model inference.

The UI should avoid overwhelming users:

- Keep raw JSON available, but move useful evidence details into structured
  rows and expandable blocks.
- Do not show full large files inline.
- Do not make planned capabilities look broken just because their future
  evidence does not exist yet.
- Continue to show the conservative ArcGIS replacement boundary clearly.

## API Design

Add a read-only endpoint:

`GET /api/ae/system/evidence`

Response shape:

```json
{
  "system": "AlphaEarth System",
  "generated_at": "2026-07-07T00:00:00Z",
  "summary": {
    "available": 3,
    "missing": 1,
    "previewable": 2,
    "blocked": 0
  },
  "checks": [
    {
      "check_id": "system:evidence_source_1_eurosat_channel_bridge",
      "capability_id": "system",
      "check_status": "pass",
      "check_title": "EuroSAT channel bridge evidence file is available",
      "evidence": [
        {
          "ref": "paper12_results/eurosat_channel_bridge_summary.json",
          "kind": "json",
          "status": "available",
          "safe_path": "paper12_results/eurosat_channel_bridge_summary.json",
          "size_bytes": 1024,
          "modified_at": "2026-07-06T00:00:00Z",
          "preview": {
            "type": "json",
            "truncated": false,
            "content": {
              "best_method": "learned_bridge_houlsby"
            }
          },
          "message": "Evidence artifact is available."
        }
      ]
    }
  ],
  "artifacts": [
    {
      "ref": "paper12_results/eurosat_channel_bridge_summary.json",
      "kind": "json",
      "status": "available",
      "safe_path": "paper12_results/eurosat_channel_bridge_summary.json",
      "previewable": true
    }
  ],
  "notes": [
    "Evidence drill-down reads local metadata and small previews only.",
    "It does not execute model inference or regenerate results."
  ]
}
```

Status semantics:

- `available`: referenced artifact exists or registry source is recognized.
- `missing`: optional artifact is absent; the system can still run.
- `blocked`: ref is outside allowed evidence roots or cannot be safely read.
- `not_applicable`: evidence ref does not represent a local artifact.

## Backend Components

Create a focused evidence service:

`ae_backend/app/services/system_evidence.py`

Responsibilities:

- Call `build_system_verification(registry)` and use its check records as the
  source of truth.
- Resolve each check's `evidence_refs`.
- Classify refs as registry refs, local artifact refs, missing artifacts, or
  blocked unsafe paths.
- Produce artifact metadata and small previews.
- Avoid PyTorch, rasterio, model loading, network access, and long-running
  operations.
- Keep all file reads bounded by size and extension.

Modify the existing system router:

- `ae_backend/app/api/system.py`
- Add `GET /evidence`
- Reuse `get_model_registry()`.

## Path Safety

Evidence drill-down must not become an arbitrary file reader.

Allowed local evidence roots:

- `paper12_results/`
- `results/`
- `linhe_results/`

Allowed registry-style refs:

- `model_hub_registry`
- `system_capabilities`

Rules:

- Normalize local refs against `PROJECT_ROOT`.
- Reject absolute paths from API output unless they resolve under an allowed
  evidence root.
- Reject `..` traversal.
- Return `blocked` for unsafe refs with a short message.
- Do not expose full absolute filesystem paths in the API response.
- Use relative `safe_path` values for display.

## Preview Rules

Preview only small, safe artifacts.

Supported preview types:

- JSON: parse and return a compact object, limited by depth and item count.
- CSV: return header and first few rows.
- TXT/MD/LOG: return first lines.

Metadata-only artifact types:

- PNG/JPG/JPEG/WebP/TIFF/TIF
- PDF
- GeoTIFF
- NPZ/PT/PTH
- Unknown binary files

Limits:

- Do not read files larger than 256 KB for inline preview.
- Return metadata for larger files.
- Mark previews as `truncated` when content is shortened.
- If parsing fails, return metadata plus a warning message rather than failing
  the whole endpoint.

## Frontend Components

Modify only `ae_frontend/index.html` in the implementation pass.

Add state and helpers:

- `systemEvidence`
- `loadingSystemEvidence`
- `systemEvidenceError`
- `fetchSystemEvidence`
- `evidenceForCheck(checkId)`
- `evidenceStatusClass(status)`
- `formatArtifactSize(bytes)`
- `formatArtifactPreview(artifact)`

Update the existing Model Hub workbench:

- Fetch evidence alongside capabilities and verification.
- Add an Evidence drill-down summary block below System verification.
- In each model card, expand check ids into check rows when evidence is
  available.
- For each check row, show artifact metadata and preview details.
- Keep existing model-hub job controls unchanged.

The UI should remain compact and operational:

- Use chips for statuses.
- Use `details` elements for previews and large JSON blocks.
- Keep text wrapping and `break-all` behavior for long paths.
- Do not add a new navigation tab in this slice.

## Testing

Use test-first implementation.

Backend tests:

- `GET /api/ae/system/evidence` returns 200.
- Response includes `summary`, `checks`, `artifacts`, and `notes`.
- Existing evidence files return `available` metadata.
- Missing optional evidence files return `missing`, not `fail`.
- Registry refs such as `model_hub_registry` return `not_applicable` or
  registry-source metadata, not missing file errors.
- Unsafe refs are blocked and do not expose absolute paths.
- JSON and CSV previews are bounded and structured.

Frontend tests:

- `ae_frontend/index.html` references `/api/ae/system/evidence`.
- Frontend state includes `systemEvidence` and `fetchSystemEvidence`.
- UI exposes `Evidence drill-down`, `evidenceForCheck`,
  `evidenceStatusClass`, `formatArtifactSize`, and preview hooks.
- Existing capability, verification, and model-hub job controls remain present.

Focused verification:

- `python -m pytest tests/test_model_hub_api.py tests/test_model_hub_frontend_entry.py`
- `python -m pytest tests/test_model_hub_registry.py tests/test_inference_api.py tests/test_inference_service.py`
- `git diff --check`

## Non-Goals

- Do not run model inference.
- Do not train models.
- Do not regenerate Paper 12 results.
- Do not edit Paper 12 manuscript files.
- Do not build a full artifact-management database.
- Do not add file upload, artifact deletion, or artifact regeneration buttons.
- Do not claim ArcGIS replacement readiness.

## Acceptance Criteria

- The backend exposes deterministic evidence drill-down data at
  `/api/ae/system/evidence`.
- The UI lets users inspect evidence behind verification checks inside the
  existing Model Hub workbench.
- Missing optional evidence is visible but non-breaking.
- Unsafe paths are blocked.
- JSON/CSV/text previews are bounded and do not expose arbitrary files.
- Focused tests pass.

## Implementation Boundary

This slice makes verification evidence inspectable. Later work can add a full
Evidence Browser, one-click regeneration jobs, checkpoint upload workflows, or
deeper Prithvi/ArcGIS comparison pages. Those should build on this read-only
evidence layer rather than bypassing it.

