# Commercial Model Hub Phase 1 Design

Date: 2026-06-26
Worktree: `D:\tmp\alphaearth-paper12-results-20260619`
Branch: `paper12-results-colab-20260619`

## Goal

Phase 1 turns the Paper 12 Prithvi/PEFT research code into a commercial-grade
remote-sensing model product line for government natural-resource and remote
sensing institute users.

The product target is an ArcGIS AI-model-like experience: a user selects a
registered model, provides imagery or an area of interest, starts an inference
job, receives geospatial result layers, and can inspect model metadata,
accuracy, applicability, and outputs without touching PyTorch code.

The first commercial package focuses on natural-resource supervision rather
than a broad clone of every ArcGIS model. The initial task family is:

1. LULC land-cover segmentation.
2. Building extraction.
3. Road and hard-surface extraction.
4. Water and flood segmentation.
5. Semantic and visual change detection.

## Design Principles

- Build a reusable model platform, not five one-off demos.
- Every model must have explicit input requirements, output schema, metrics,
  versioning, and example data.
- Raster inference must work at map scale through tiling, stitching, and GIS
  export, not only on isolated patches.
- Paper 12 assets remain useful: Prithvi-100M, PEFT adapters, Linhe LULC,
  LoveDA/LandCover.ai segmentation, and the existing LULC API become the base.
- Phase 1 should produce a stable government-demo path before expanding to
  agriculture, city management, point cloud, text, or time-series products.

## Non-Goals

- Phase 1 does not implement every ArcGIS pretrained model category.
- Phase 1 does not promise universal global accuracy. Each model declares its
  trained region, sensor assumptions, and supported input format.
- Phase 1 does not require a new foundation model. It uses Prithvi-100M and
  existing PEFT/task heads first, with space for later backbones.
- Phase 1 does not require full online marketplace billing or tenant isolation.
  It prepares metadata and job boundaries that can support those later.

## Existing Assets To Reuse

- `geoadapter.models.PrithviBackbone`: local Prithvi-100M backbone wrapper.
- `geoadapter.models.heads`: classification, multilabel, and segmentation heads.
- `geoadapter.bench.run_benchmark`: YAML-driven PEFT training/evaluation runner.
- `ae_backend.app.services.inference`: current LULC segmentation inference
  service for Prithvi + adapter + segmentation head checkpoints.
- `ae_backend.app.api.inference`: existing `/api/ae/inference/lulc` endpoints.
- `scripts/linhe_change_detect.py`: current RGB-diff and PCA-RX change pipeline.
- `linhe_results/linhe_lulc_seg.json`: Linhe LULC method metrics for product
  cards and model-ranking defaults.
- `results/linhe_change/`: existing change heatmap and pair visual outputs.

## Phase 1 Model Catalog

### `lulc_6class_prithvi_houlsby`

- Task: semantic segmentation.
- Classes: background, built, crops, trees, water, rangeland/bare.
- Input: RGB patch or tiled RGB raster, bridged to Prithvi six-channel input.
- Backbone: Prithvi-100M.
- Adaptation: Houlsby first, GeoAdapter as a lightweight alternative.
- Output: class raster, color preview PNG, class-area statistics, optional
  polygonized GeoJSON.
- Current evidence: Linhe LULC results already show Houlsby as the strongest
  method and GeoAdapter as a useful lightweight option.

### `building_extraction_prithvi`

- Task: binary segmentation or instance-candidate extraction.
- Classes: background, building.
- Input: RGB high-resolution imagery.
- Training data: existing Linhe OSM building masks as the first weak-supervision
  source, with later manual validation samples.
- Output: binary raster, building polygons, footprint area, patch-level summary.
- Risk: OSM labels are incomplete; production claims must distinguish weak-label
  extraction from independently validated building detection.

### `road_hardscape_prithvi`

- Task: semantic segmentation.
- Classes: background, road/hard-surface.
- Input: RGB high-resolution imagery.
- Training data: OSM roads and/or locally provided vector roads rasterized onto
  the Linhe patch grid.
- Output: segmentation raster, vector centerline or polygon candidates, length
  and area statistics.
- Risk: road width and occlusion require postprocessing and validation.

### `water_flood_prithvi`

- Task: water segmentation, with flood mode when before/after or seasonal
  reference imagery is available.
- Input: RGB or multispectral imagery where available.
- Output: water mask, flood-candidate mask, water-area statistics, change report.
- Risk: RGB-only water detection is weaker than multispectral/SAR water mapping;
  metadata must state supported sensors.

### `semantic_change_prithvi`

- Task: change detection.
- Inputs: paired same-area rasters or paired Linhe-style patches.
- Channels:
  - Visual change: RGB L2 difference and PCA-RX anomaly score, reusing the
    current change pipeline.
  - Semantic change: compare LULC/building/water predictions across dates.
- Output: change heatmap, changed polygons, from-to class matrix, top changed
  patches, before/after preview images.
- Phase 1 status: not a Prithvi embedding-difference detector yet. It is a
  productized combination of visual anomaly detection and semantic map
  differencing. A later phase can add Prithvi feature-distance change scoring.

## Architecture

### Model Registry

Add a registry layer under the backend. The first implementation can be a JSON
or YAML registry committed in the repository; it should be shaped so it can move
to PostGIS or another database later.

Required fields:

- `model_id`
- `display_name`
- `task_type`
- `backbone`
- `adapter`
- `checkpoint_path`
- `input_spec`
- `output_spec`
- `class_schema`
- `metrics`
- `trained_region`
- `supported_sensors`
- `license`
- `status`
- `example_inputs`
- `created_at`
- `updated_at`

### Unified Job API

Add a model-hub API namespace:

- `GET /api/ae/model-hub/models`
- `GET /api/ae/model-hub/models/{model_id}`
- `POST /api/ae/model-hub/jobs`
- `GET /api/ae/model-hub/jobs/{job_id}`
- `GET /api/ae/model-hub/jobs/{job_id}/results`
- `GET /api/ae/model-hub/jobs/{job_id}/logs`

Job creation accepts:

- `model_id`
- input mode: uploaded raster, uploaded image patch, cached dataset, or bbox
- optional date/date-pair for change detection
- output formats: PNG, GeoTIFF, GeoJSON, CSV
- inference options: tile size, stride, device, confidence threshold

### Raster Processing Pipeline

The raster pipeline is shared by all imagery models:

1. Validate CRS, transform, bands, pixel size, and nodata.
2. Normalize input into the model's declared `input_spec`.
3. Tile large rasters with overlap.
4. Run model inference per tile.
5. Stitch logits or class masks.
6. Apply optional smoothing and small-object filtering.
7. Export class raster as GeoTIFF or COG-compatible GeoTIFF.
8. Export preview PNG tiles for frontend.
9. Polygonize selected classes when vector output is requested.
10. Compute class areas, counts, and change statistics.

### Inference Runtime

The runtime wraps task-specific predictors behind one interface:

- `predict_patch(input) -> patch_result`
- `predict_raster(input_raster, options) -> raster_result`
- `summarize(result) -> metrics/statistics`
- `export(result, formats) -> file manifest`

Existing LULC inference becomes the first concrete implementation. Other models
reuse the same adapter/backbone/head loading pattern.

### Frontend

Add a "Model Hub" section:

- model cards grouped by task type
- model detail panel with input/output requirements and metrics
- run form for upload, bbox, or cached demo data
- job progress panel
- map result viewer for raster preview and GeoJSON overlays
- statistics panel with area tables and change matrices
- download panel for generated files

The frontend should not expose implementation details such as PEFT training
internals unless the user opens an advanced details panel.

## Data Flow

### Single-Date Segmentation

User input raster -> validation -> tiling -> model inference -> stitched mask ->
class statistics -> GeoTIFF/PNG/GeoJSON export -> frontend map and downloads.

### Two-Date Change Detection

Date A raster + Date B raster -> co-registration check -> visual change scoring
-> per-date semantic inference -> class-map differencing -> changed polygons ->
from-to matrix -> report and map outputs.

## Testing Strategy

Unit tests:

- registry parsing and validation
- model lookup and unsupported-model errors
- input-spec validation
- tile grid generation
- patch stitching shape invariants
- area-statistics correctness
- API contract tests for models, jobs, and results

Integration tests:

- local LULC patch inference through the model-hub API
- small synthetic raster tiled through the shared pipeline
- change-detection job using two tiny paired rasters
- missing checkpoint and missing Prithvi-weight error paths

Artifact tests:

- exported GeoTIFF preserves transform and CRS
- GeoJSON polygons contain required properties
- CSV summaries include model id, job id, class names, and areas

## Acceptance Criteria

Phase 1 is complete when:

- The backend exposes a model catalog with at least five phase-1 model entries.
- At least two entries are runnable end-to-end: LULC segmentation and change
  detection.
- A user can start an inference job without writing Python.
- Results include preview PNG and statistics for all runnable jobs.
- LULC raster outputs include GIS-preserving GeoTIFF export.
- Change detection outputs a heatmap, top changed patches, and a class-change
  summary when semantic inputs are available.
- The frontend has a Model Hub page that can list models, start a demo job, and
  display result artifacts.
- The system reports clear errors for missing weights, unsupported bands,
  invalid CRS, and unavailable model checkpoints.
- Tests cover the model registry, API contracts, raster tiling, output
  statistics, and the two initial runnable jobs.

## Milestones

### Milestone 1: Registry And API Skeleton

- Add model registry schema and sample phase-1 entries.
- Add model-hub API routes for model listing and model details.
- Add job model and in-memory job lifecycle for local development.
- Add tests for registry and API contracts.

### Milestone 2: LULC As First Productized Model

- Wrap the existing LULC inference service behind the model-hub runtime.
- Add patch-level and raster-level LULC inference modes.
- Export PNG preview, class table, and GeoTIFF mask.
- Add frontend model card and run form for LULC.

### Milestone 3: Shared Raster Pipeline

- Implement tiling, stitching, CRS-preserving export, and statistics.
- Reuse it for LULC and prepare it for the next segmentation models.
- Add synthetic-raster integration tests.

### Milestone 4: Change Detection Productization

- Wrap the existing Linhe change outputs and PCA-RX code as a model-hub job.
- Add semantic differencing when date-pair class maps are available.
- Export heatmap, changed polygons, top-patch list, and summary CSV.

### Milestone 5: Additional Natural-Resource Model Entries

- Add building, road/hard-surface, and water/flood model entries.
- Initially mark entries as `planned` or `demo_only` unless runnable checkpoints
  and validation metrics are present.
- Add training/fine-tuning hooks only after the inference product path is stable.

## Risks And Mitigations

- Missing pretrained weights or checkpoints: expose readiness status in the
  registry and fail early with actionable errors.
- Weak labels for building and road extraction: label models as weak-supervised
  until manual validation is complete.
- RGB-only limitations: model metadata must declare supported sensors and
  expected accuracy boundaries.
- Large raster memory pressure: tile by default and cap maximum in-memory arrays.
- Product sprawl: new models must enter through the registry and shared runtime,
  not through ad hoc endpoints.

## Approved Boundary

The approved Phase 1 direction is:

- customer priority: government natural-resource and remote sensing institute
  users;
- strategy: domestic business landing plus Model Hub platform foundation;
- first runnable capabilities: LULC segmentation and change detection;
- next entries: building, road/hard-surface, and water/flood models;
- no code implementation starts until this spec is reviewed and accepted.

## References

- ArcGIS AI models overview:
  https://doc.arcgis.com/en/pretrained-models/latest/get-started/intro.htm
- ArcGIS Prithvi Crop Classification reference:
  https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-prithvi-crop-classification.htm
