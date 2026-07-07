# Production Model Realization M1 Design

Date: 2026-07-07
Workspace: `D:\adk\AlphaEarth-System`
Branch: `master`

## Goal

M1 turns the current Model Hub from a mixed research/demo registry into a
truthful production-readiness system. A model may be shown as production-ready
only when the system can verify all of the following:

- a real checkpoint or external pretrained model package is configured;
- a public or local test sample can run end to end;
- preprocessing, tiling, stitching, and GIS artifact export are implemented;
- validation metrics and limitations are visible in the UI and API;
- the runtime logs prove whether a neural checkpoint, a cached artifact, or a
  training/evaluation placeholder was used.

The immediate production candidates are crop classification, flood mapping, and
the existing LULC checkpoints. Building, road/hardscape, and semantic change
detection enter M1 as real data/training pipelines first, not as fake ready
models.

## Current Reality

The committed registry currently mixes three levels:

- `ready`: `lulc_6class_prithvi_houlsby`, backed by a local Houlsby checkpoint.
- `demo_only`: crop and semantic change, where crop raster mode currently uses
  deterministic rules and semantic change uses cached Linhe PCA-RX outputs.
- `planned`: building, road/hardscape, and water/flood entries that do not have
  runnable checkpoints or real runtimes wired through Model Hub.

This must be made explicit. M1 does not rename placeholders to `ready`. It adds
verification gates so the system can only promote a model when evidence exists.

## Public Sources

M1 uses public sources that have stable model cards or data pages:

- IBM/NASA Prithvi-EO-1.0-100M, Apache-2.0, pretrained on HLS six-band temporal
  imagery and accepting `(B, C, T, H, W)` HLS-style inputs.
- IBM/NASA Prithvi-EO-1.0-100M multi-temporal crop classification, Apache-2.0,
  trained for 224 x 224 x 18 HLS chips and 13 CDL-derived classes.
- IBM/NASA multi-temporal crop classification dataset, CC-BY-4.0, 3,854 chips,
  train/validation split, 18-band HLS GeoTIFF inputs and single-band masks.
- IBM/NASA Prithvi-EO-1.0-100M Sen1Floods11, Apache-2.0, six-band Sentinel-2
  flood mapping checkpoint with reported mIoU and inference script.
- Sen1Floods11 public dataset for flood mapping.
- SpaceNet open data on AWS for building, road, flood, and multi-temporal urban
  development tasks.
- Microsoft Global ML Building Footprints, CDLA Permissive 2.0, for footprint
  labels and independent building validation.

## Scope

### M1A: Readiness And Evidence Layer

Add a production evidence layer for Model Hub:

- `weights`: expected checkpoint source, local path, optional Hugging Face repo,
  required files, and checksum if known.
- `test_data`: public source, local cache root, sample input paths, label paths,
  license, and expected shape/band order.
- `verification`: last local verification status, command, timestamp, metrics,
  and artifacts.
- `runtime_kind`: one of `neural_checkpoint`, `public_product`,
  `cached_artifact`, `training_pipeline`, or `contract_demo`.
- `promotion_policy`: `ready` requires a runnable checkpoint and at least one
  passing local inference test; `production_candidate` additionally requires
  documented validation metrics on non-synthetic data.

The frontend must display these fields in Chinese and must not hide missing
weights or missing datasets.

### M1B: Real Crop Runtime Track

Replace the current crop raster deterministic classification as the default
truth path with a real Prithvi crop runtime when the public IBM/NASA package is
available locally.

Required behavior:

- keep `upload_raster_demo` available for contract checks, but label it as
  `contract_demo`;
- add a `real_raster_inference` or equivalent mode for neural inference;
- validate 18-band HLS GeoTIFF input: three time steps, each ordered Blue,
  Green, Red, Narrow NIR, SWIR 1, SWIR 2;
- load the IBM/NASA crop model through a supported runtime path, initially
  TerraTorch/mmseg-compatible if the environment supports it, otherwise a
  clearly failed readiness state with install guidance;
- export classified GeoTIFF, PNG preview, GeoJSON polygons, CSV summary, and
  manifest;
- verify on at least one public crop validation chip or small cached sample.

If the real runtime dependencies are unavailable, the model remains
`download_required` or `dependency_required`, not `ready`.

### M1C: Real Flood Runtime Track

Promote water/flood from `planned` by wiring the IBM/NASA Sen1Floods11 Prithvi
checkpoint as the first real water/flood model.

Required behavior:

- add six-band Sentinel-2 GeoTIFF validation in the order Blue, Green, Red,
  Narrow NIR, SWIR 1, SWIR 2;
- support binary class schema: no water, water/flood, with nodata/cloud handling
  when labels use -1;
- load the public checkpoint through a supported runtime path;
- run at least one public Sen1Floods11 sample end to end;
- export GeoTIFF, PNG, GeoJSON, CSV, and manifest;
- register reported model-card metrics and local smoke-test metrics separately.

### M1D: LULC Production Wrapper

The LULC checkpoints are real, but the production wrapper is still incomplete.
M1 adds:

- patch and raster inference modes with explicit GeoTIFF output;
- the same artifact manifest shape as crop/flood;
- a public/local test input list tied to real Linhe patch data;
- readiness checks for both GeoAdapter and Houlsby checkpoints.

### M1E: Building, Road, And Change Data Pipelines

These models should not become `ready` in M1 unless training completes and a
checkpoint passes inference verification.

M1 adds data and training scaffolds:

- Building: SpaceNet building samples and/or Microsoft building footprint label
  acquisition, rasterization, split manifests, and evaluation hooks.
- Road/hardscape: SpaceNet road graph/road mask conversion, split manifests,
  training config, and metric reporting.
- Semantic change: SpaceNet 7 or another public change dataset adapter,
  two-date raster validation, change mask or building-change labels, and
  evaluation hooks.

Their registry status becomes `training_data_ready` or `training_required` only
after data acquisition tests pass.

## Non-Goals

- Do not commit large downloaded model weights or public datasets to git.
- Do not claim ArcGIS `.dlpk` compatibility unless a `.dlpk` import/export path
  is explicitly implemented and tested.
- Do not claim global production accuracy from a smoke test.
- Do not silently fall back from real inference to deterministic demo output.
- Do not require online downloads during ordinary unit tests.

## Backend Architecture

Add or extend these components:

- `model_hub_evidence.py`: builds per-model evidence from registry, filesystem,
  weights manifests, data manifests, and latest verification records.
- `model_asset_registry.py`: declares downloadable public weights and datasets
  without downloading them automatically.
- `public_data_sources.json`: source URLs, licenses, expected file layouts, and
  sample selectors.
- `model_hub_real_crop.py`: real crop runtime wrapper, input validator, and
  artifact writer.
- `model_hub_flood.py`: real flood runtime wrapper, input validator, and
  artifact writer.
- `model_hub_lulc_raster.py`: raster-level wrapper around existing LULC
  checkpoints.
- CLI scripts under `scripts/model_hub/`:
  - `verify_assets.py`
  - `fetch_public_sample.py`
  - `verify_model_runtime.py`

Network downloads must be explicit commands, not implicit imports.

## API Contract

Extend the existing APIs rather than adding unrelated endpoints:

- `GET /api/ae/model-hub/models`: include status, runtime kind, evidence
  summary, and supported input modes.
- `GET /api/ae/system/evidence`: include real weight/data checks and last
  verification results.
- `POST /api/ae/model-hub/jobs`: reject real inference modes when weights,
  dependencies, or test data are missing; do not create long-lived pending jobs
  for impossible runs.

The job response must include:

- `runtime_kind`;
- `model_source`;
- `input_validation`;
- `artifacts`;
- `limitations`;
- `logs` with enough detail to distinguish real inference from demo logic.

## Frontend

The Model Hub UI must show:

- production-ready, dependency-required, download-required, training-required,
  demo-only, and planned states with different Chinese labels;
- which public dataset or checkpoint backs each model;
- whether the current local machine has the required files;
- direct test-data paths for local trials;
- a clear warning when a job uses `contract_demo` or cached outputs.

No model card may present demo output as production inference.

## Testing

Use unit tests without network access and mark dataset downloads as explicit
integration tasks.

Required tests:

- registry/evidence tests prove a model cannot become `ready` without real
  checkpoint evidence;
- missing dependency and missing weight paths fail clearly;
- crop 18-band validation accepts public-style chips and rejects wrong band
  counts;
- flood six-band validation accepts public-style chips and rejects wrong band
  counts;
- LULC raster wrapper preserves CRS, transform, and dimensions;
- frontend tests assert Chinese labels for production states and warnings;
- CLI dry-run tests emit planned download commands without touching the network.

Manual/integration verification:

- crop: one public crop validation chip through real runtime, if dependencies
  and weights are installed;
- flood: one Sen1Floods11 sample through real runtime, if dependencies and
  weights are installed;
- LULC: existing Linhe RGB patch/raster through local checkpoints.

## Acceptance Criteria

M1 is complete when:

- Model Hub shows truthful production readiness for every registered model.
- Crop and flood have real public checkpoint integration paths and cannot
  report `ready` unless their local inference tests pass.
- LULC has production-style artifact export beyond raw JSON masks.
- Building, road, and semantic change expose public data acquisition/training
  pipelines without being mislabeled as production models.
- The UI gives users usable test data paths and source provenance.
- Focused tests and full pytest pass.
- `git diff --check` is clean.

## Risks

- Public Prithvi runtimes may require TerraTorch/mmseg versions that conflict
  with the current lightweight environment. Mitigation: isolate optional
  dependencies and fail with actionable readiness diagnostics.
- Public datasets are large. Mitigation: support sample-only downloads and keep
  large data outside git.
- Model-card metrics and local smoke metrics are not equivalent. Mitigation:
  store them separately and label them in the UI.
- Building/road/change may require training time. Mitigation: use M1 to make
  the data and training path real, then promote only after checkpoints exist.

## References

- ArcGIS Prithvi Crop Classification documentation:
  https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-prithvi-crop-classification.htm
- IBM/NASA Prithvi-EO-1.0-100M:
  https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- IBM/NASA Prithvi crop classification model:
  https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification
- IBM/NASA multi-temporal crop classification dataset:
  https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification
- IBM/NASA Prithvi Sen1Floods11 model:
  https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11
- Sen1Floods11 dataset:
  https://github.com/cloudtostreet/Sen1Floods11
- SpaceNet open data on AWS:
  https://registry.opendata.aws/spacenet/
- Microsoft Global ML Building Footprints:
  https://github.com/microsoft/GlobalMLBuildingFootprints
