# Paper12 ArcGIS Replacement Evaluator Design

## Goal

Add the next concrete step after the conservative replacement boundary: a reproducible Linhe validation manifest schema and evaluator that can compare manual labels, ArcGIS/Esri reference outputs, and Paper12 outputs on the same samples.

## Confirmed Scope

The evaluator does not create manual labels, run ArcGIS, or run Paper12 inference. It consumes paired outputs that already exist. Until those inputs are present, the official status remains `not_validated`.

## Input Contract

The validation manifest is a CSV with one row per validation sample. Required columns are:

- `sample_id`
- `manual_mask_path` or `manual_label`
- `arcgis_mask_path` or `arcgis_label`
- `paper12_mask_path` or `paper12_label`

Optional columns such as `scene_id`, `x`, `y`, `dominant_esri_class`, `dominant_paper12_class`, `annotator_id`, and `review_status` are allowed and preserved only as sample metadata. Mask paths are resolved relative to the manifest file. The first implementation supports `.npy`, `.npz`, `.csv`, and, when rasterio is installed, raster masks.

## Metrics

The evaluator reports metrics for ArcGIS vs manual and Paper12 vs manual:

- overall accuracy,
- macro F1,
- per-class IoU,
- mIoU,
- confusion matrix.

The paired delta is `paper12 - arcgis` for overall accuracy, macro F1, and mIoU.

## Decision Rule

The output status is:

- `not_validated` when required paired evidence is missing.
- `replacement_candidate` when Paper12 mIoU is not below ArcGIS by more than the configured tolerance and all critical classes meet the configured per-class tolerance.
- `partial` when paired evidence exists but the candidate rule fails.

Default critical classes are `water`, `crops`, and `built`; default tolerances are `0.0`, meaning Paper12 must match or exceed ArcGIS.

## Files

- `paper12_results/linhe_manual_validation_protocol.json`: protocol and manifest schema.
- `paper12_results/linhe_manual_validation_manifest_template.csv`: header-only manifest template.
- `scripts/evaluate_arcgis_replacement.py`: CLI and importable evaluator.
- `tests/test_arcgis_replacement_evaluator.py`: unit tests for metrics and decision rules.
- Supplementary mirrors under `submission/paper12_isprs_jprs_20260606/06_supplementary_material/paper12_results/`.
