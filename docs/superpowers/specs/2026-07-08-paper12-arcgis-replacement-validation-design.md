# Paper12 ArcGIS Replacement Validation Design

## Goal

Define a conservative, testable boundary for whether Paper12's Prithvi/Houlsby workflow can be treated as an ArcGIS-style replacement. The current evidence supports a local weak-supervision adaptation workflow, not a production replacement claim.

## Decision Rule

The replacement status remains `not_validated` until all of the following evidence is available for the same area, time window, and class taxonomy:

- independent manual ground truth,
- an ArcGIS or Esri reference model output,
- a Paper12 checkpoint-backed output,
- per-class and aggregate metrics comparing both model outputs against the same manual labels.

If those conditions are met and Paper12 matches or exceeds the ArcGIS reference on the primary metrics without unacceptable per-class regressions, the status can move to `replacement_candidate`. Otherwise it remains `partial` or `not_validated`.

## Scope

This change adds no fabricated validation results. It adds a committed validation template, audit fields, system-summary reporting, manuscript guardrails, and tests that prevent overclaiming.

## Files

- `paper12_results/arcgis_replacement_validation_template.json`: machine-readable validation protocol and empty result slots.
- `geoadapter/bench/paper12_audit.py`: audit derivation for replacement readiness.
- `paper12_results/review_audit_summary.json`: regenerated derived audit.
- `ae_backend/app/services/paper12_summary.py`: exposes the validation boundary in the model hub summary.
- `ae_backend/app/services/system_capabilities.py`: includes the optional validation template as evidence.
- Paper12 manuscript and submission text: clarify that ArcGIS replacement is not yet validated.
- Tests under `tests/`: guard the template, audit, backend summary, and manuscript wording.
