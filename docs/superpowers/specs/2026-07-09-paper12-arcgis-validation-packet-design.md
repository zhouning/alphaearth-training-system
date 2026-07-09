# Paper12 ArcGIS Validation Packet Design

## Goal

Prepare the missing step between the conservative ArcGIS replacement boundary
and the evaluator: a reproducible Linhe annotation packet that selects existing
patches, exports Esri reference masks, and creates a manifest stub for manual
truth and Paper12 outputs.

## Boundary

The packet builder does not create manual labels, run ArcGIS, or run Paper12
inference. Its output remains `evaluator_ready: false` until the manifest has
both independent manual masks and checkpoint-backed Paper12 masks.

## Inputs

The builder reads a Linhe LULC index from CSV or parquet. Required columns are
`patch_path` and `lulc_path`; optional metadata such as `sample_id`, `scene_id`,
`year`, and patch coordinates are preserved in the annotation manifest. Parquet
indexes require `pyarrow` or `fastparquet`; CSV input is used by tests and works
without extra engines.

The builder also supports `--index auto`, which scans a Linhe patch root for
`*/p_*.npz` and matching `*/lulc_<year>_p_*.npz` files. This mode avoids the
local parquet-reader dependency and works with the checked-out Linhe patch tree.

## Outputs

The packet directory contains:

- `arcgis_replacement_annotation_manifest.csv` with blank manual and Paper12
  evidence fields,
- `rgb/*.npy` and `arcgis_masks/*.npy`,
- empty target directories for `manual_masks/` and `paper12_masks/`,
- `previews/*.png` for human review,
- `annotation_readme.md`,
- `packet_summary.json`.

## Sampling

Sampling is deterministic by seed. The builder first tries to cover requested
critical classes (`water`, `crops`, `built` by default) using Esri mask class
fractions, then fills remaining slots by seeded shuffle.

## Validation

Tests cover CSV input, conservative output status, critical-class coverage,
preview and manifest creation, CLI execution, filesystem auto-discovery, lazy
RGB loading, and protocol links to the packet builder. A local smoke run on
`data/linhe_patches` generated a six-sample packet under `D:\tmp` and the
evaluator correctly kept it `not_validated` because manual and Paper12 masks
remain missing.
