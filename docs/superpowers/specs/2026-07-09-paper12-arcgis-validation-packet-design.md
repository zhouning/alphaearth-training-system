# Paper12 ArcGIS Validation Packet Design

## Goal

Prepare the missing steps between the conservative ArcGIS replacement boundary
and the evaluator: a reproducible Linhe annotation packet that selects existing
patches, exports Esri reference masks, creates a manifest stub for manual truth
and Paper12 outputs, and finalizes completed packets only after evidence files
exist and pass shape checks.

## Boundary

The packet builder and finalizer do not create manual labels, run ArcGIS, or run
Paper12 inference. Builder output remains `evaluator_ready: false`; the
finalizer only produces an evaluator manifest after independent manual masks and
checkpoint-backed Paper12 masks exist for every packet sample and match the Esri
mask shapes.

## Inputs

The builder reads a Linhe LULC index from CSV or parquet. Required columns are
`patch_path` and `lulc_path`; optional metadata such as `sample_id`, `scene_id`,
`year`, and patch coordinates are preserved in the annotation manifest. Parquet
indexes require `pyarrow` or `fastparquet`; CSV input is used by tests and works
without extra engines.

The builder also supports `--index auto`, which scans a Linhe patch root for
`*/p_*.npz` and matching `*/lulc_<year>_p_*.npz` files. This mode avoids the
local parquet-reader dependency and works with the checked-out Linhe patch tree.

The finalizer reads `arcgis_replacement_annotation_manifest.csv` from a packet
directory and checks only existing `manual_masks/`, `paper12_masks/`, and
`arcgis_masks/` files.

## Outputs

The packet directory contains:

- `arcgis_replacement_annotation_manifest.csv` with blank manual and Paper12
  evidence fields,
- `rgb/*.npy` and `arcgis_masks/*.npy`,
- empty target directories for `manual_masks/` and `paper12_masks/`,
- `previews/*.png` for human review,
- `annotation_readme.md`,
- `packet_summary.json`.

When completed evidence is present, the finalizer writes:

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
RGB loading, finalizer missing-evidence handling, evaluator-manifest creation,
shape mismatch rejection, and protocol links to the packet builder/finalizer. A
local smoke run on `data/linhe_patches` generated a six-sample packet under
`D:\tmp` and the evaluator correctly kept it `not_validated` because manual and
Paper12 masks remain missing.