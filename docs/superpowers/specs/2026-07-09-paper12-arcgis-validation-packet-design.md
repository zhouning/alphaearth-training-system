# Paper12 ArcGIS Validation Packet Design

## Goal

Prepare the missing steps between the conservative ArcGIS replacement boundary
and the evaluator: a reproducible Linhe annotation packet that selects existing
patches, exports Esri reference masks, creates a manifest stub for manual truth
and Paper12 outputs, audits packet readiness, exports checkpoint-backed Paper12
predictions, and finalizes completed packets only after evidence files exist and
pass shape checks.

## Boundary

The packet builder, readiness audit, prediction exporter, and finalizer do not
create manual labels, run ArcGIS, or evaluate replacement status. Builder output
remains `evaluator_ready: false`; the audit writes only diagnostic readiness
summaries and next actions; the exporter only writes Paper12 checkpoint outputs
to `paper12_masks/`; the finalizer only produces an evaluator manifest after
independent manual masks and checkpoint-backed Paper12 masks exist for every
packet sample and match the Esri mask shapes.

## Inputs

The builder reads a Linhe LULC index from CSV or parquet. Required columns are
`patch_path` and `lulc_path`; optional metadata such as `sample_id`, `scene_id`,
`year`, and patch coordinates are preserved in the annotation manifest. Parquet
indexes require `pyarrow` or `fastparquet`; CSV input is used by tests and works
without extra engines.

The builder also supports `--index auto`, which scans a Linhe patch root for
`*/p_*.npz` and matching `*/lulc_<year>_p_*.npz` files. This mode avoids the
local parquet-reader dependency and works with the checked-out Linhe patch tree.

The readiness audit reads the packet manifest and existing mask files to report
whether the packet is waiting for manual labels, waiting for Paper12 predictions,
ready for finalization, ready for evaluation, or blocked by shape errors. The
prediction exporter reads packet RGB chips and a benchmark-style Paper12 LULC
segmentation checkpoint. The finalizer reads
`arcgis_replacement_annotation_manifest.csv` from a packet directory and checks
only existing `manual_masks/`, `paper12_masks/`, and `arcgis_masks/` files.

## Outputs

The packet directory contains:

- `arcgis_replacement_annotation_manifest.csv` with blank manual and Paper12
  evidence fields,
- `rgb/*.npy` and `arcgis_masks/*.npy`,
- empty target directories for `manual_masks/` and `paper12_masks/`,
- `previews/*.png` for human review,
- `annotation_readme.md`,
- `packet_summary.json`.

At any point, the readiness audit can write:

- `packet_readiness_summary.json` with status, evidence counts, missing-evidence
  rows, shape errors, and next-action commands.

When a Paper12 checkpoint is supplied, the prediction exporter writes:

- `paper12_masks/<sample_id>.npy` for each exported sample,
- `paper12_prediction_export_summary.json` with exported, skipped, and failed
  sample diagnostics.

When completed manual and Paper12 evidence is present, the finalizer writes:

- `arcgis_replacement_evaluator_manifest.csv` with evaluator-compatible columns,
- `packet_finalization_summary.json` with missing-evidence and shape-error
  diagnostics.

## Sampling

Sampling is deterministic by seed. The builder first tries to cover requested
critical classes (`water`, `crops`, `built` by default) using Esri mask class
fractions, then fills remaining slots by seeded shuffle.

## Validation

Tests cover CSV input, conservative output status, critical-class coverage,
preview and manifest creation, CLI execution, filesystem auto-discovery, lazy
RGB loading, readiness statuses and next actions, prediction export,
resume-safe skip/overwrite behavior, prediction shape mismatch rejection,
finalizer missing-evidence handling, evaluator-manifest creation, finalizer
shape mismatch rejection, and protocol links to the packet builder/audit/exporter/finalizer.
A local smoke run on `data/linhe_patches` generated a six-sample packet under
`D:\tmp` and the evaluator correctly kept it `not_validated` because manual and
Paper12 masks remain missing.