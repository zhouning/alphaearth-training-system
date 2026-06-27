# Prithvi Crop Raster Phase 3 Design

Date: 2026-06-27
Worktree: `D:\tmp\alphaearth-paper12-results-20260619`
Branch: `paper12-prithvi-raster-phase3`

## Goal

Phase 3 moves the Paper 12 ArcGIS-style Prithvi crop package from a cached
Model Hub demo contract to a real raster-input execution contract. A user should
be able to point the Model Hub crop package at a local 18-band GeoTIFF/HLS
composite, receive clear input validation, run a deterministic tiled
classification demo, and inspect GIS-shaped outputs: classified raster, preview,
vector summary, CSV summary, and a package manifest.

This phase still does not claim production crop classification accuracy. It
implements the parts of the ArcGIS Prithvi workflow that can be made stable in
this repository without downloading large weights or requiring GPU execution.

## External Product Boundary

The target commercial analogue is Esri's ArcGIS AI model page for "Prithvi -
Crop Classification" and IBM/NASA's HuggingFace model card for
`prithvi-100m-multi-temporal-crop-classification`.

The relevant official constraints are:

- Input is an 18-band composite raster, mosaic dataset, or image service.
- Output is a classified raster with 13 crop/land-cover classes.
- The model packages the IBM/NASA Prithvi crop classification model.
- GPU execution is recommended by ArcGIS for the real workflow.
- The HuggingFace model card defines the tensor shape as `224 x 224 x 18`.
- The 18 bands are 3 time steps, each ordered as Blue, Green, Red, Narrow NIR,
  SWIR 1, and SWIR 2.

Phase 3 uses these as product-contract requirements. It does not embed Esri's
`.dlpk`, ArcPy, ArcGIS Pro, Terratorch, or real checkpoint inference.

## Product Delta From Phase 2

Phase 2 added:

- Model Hub discovery for `prithvi_crop_classification_arcgis_style`.
- ArcGIS-style model package metadata.
- `cached_demo` runtime.
- Static artifact manifests.

Phase 3 adds:

- `upload_raster_demo` runtime mode for the crop model.
- Local GeoTIFF validation against the 18-band Prithvi crop input contract.
- Tiled deterministic classification over the actual raster dimensions.
- Classified raster and manifest artifacts written under `results/model_hub`.
- More faithful crop class schema aligned with the IBM/NASA 13-class model card.
- Frontend affordance for selecting the crop raster demo mode and passing a
  local raster path.

## Non-Goals

- Do not download or commit Prithvi crop checkpoint weights.
- Do not add ArcGIS Pro, ArcPy, Esri `.dlpk`, Terratorch, or mmseg runtime
  dependencies.
- Do not run true neural network inference.
- Do not make global or production crop-classification accuracy claims.
- Do not add asynchronous background workers.
- Do not support public HTTP multipart uploads in this phase. The API accepts a
  trusted local `options.raster_path` so tests and local demos remain simple.
- Do not convert mosaic datasets or image services. This phase is local GeoTIFF
  only.

## Model Registry Contract

Update the crop model entry in `ae_backend/app/data/model_hub_models.json`.

Required changes:

- Add `upload_raster_demo` to `package_profile.runtime_modes`.
- Set `input_spec.default_demo_input_mode` to `cached_demo` so the existing
  quick demo remains one click.
- Add `input_spec.supported_job_input_modes` as
  `["cached_demo", "upload_raster_demo"]`.
- Replace or supplement the Phase 2 local crop class schema with the official
  13-class Prithvi crop classes:
  1. `natural_vegetation`
  2. `forest`
  3. `corn`
  4. `soybeans`
  5. `wetlands`
  6. `developed_barren`
  7. `open_water`
  8. `winter_wheat`
  9. `alfalfa`
  10. `fallow_idle_cropland`
  11. `cotton`
  12. `sorghum`
  13. `other`
- Add `package_profile.input_profile.band_order` with 18 named slots:
  `t1_blue`, `t1_green`, `t1_red`, `t1_narrow_nir`, `t1_swir1`, `t1_swir2`,
  repeated for `t2_*` and `t3_*`.
- Add `package_profile.output_profile.artifacts`:
  `classified_raster_geotiff`, `preview_png`, `crop_polygons_geojson`,
  `area_summary_csv`, `arcgis_style_manifest_json`.
- Add `package_profile.applicability` notes that the real published model is
  expected to work well in the United States, while this repository's Phase 3
  deterministic runtime is a contract demo only.

Compatibility note: if existing tests or frontend display logic expect the old
Phase 2 names such as `maize` or `rice`, they should be updated to use the
official 13-class schema. The old names were temporary Phase 2 demo labels and
are less accurate for the ArcGIS/Prithvi target.

## Backend Components

### 1. Raster Input Validator

Create `ae_backend/app/services/model_hub_crop_raster.py`.

Public entry points:

```python
def validate_prithvi_crop_raster(raster_path: str | Path) -> dict:
    """Validate a local GeoTIFF against the Prithvi crop raster contract."""


def run_prithvi_crop_raster_demo(*, options: dict) -> dict:
    """Run deterministic tiled crop classification over a validated raster."""
```

Validation behavior:

- Resolve `raster_path` from `options.raster_path`.
- Require the path to exist.
- Require suffix `.tif` or `.tiff`.
- Open with rasterio.
- Require `src.count == 18`.
- Require positive width and height.
- Require `src.crs` and `src.transform` to be present.
- Capture dtype, nodata values, bounds, CRS, transform, width, height, and band
  count.
- Report the fixed band order in the validation response.
- Raise `ModelHubRuntimeError` for user-facing validation failures.

The validator should not read the full raster into memory. It reads only
metadata and, during classification, tile windows.

### 2. Deterministic Tiled Classifier

The classifier is a product-contract runtime, not a neural model.

Implementation rules:

- Default tile size: `224`.
- Default stride: `224`.
- Use `make_tile_grid` from `ae_backend/app/services/raster_pipeline.py`.
- For each tile, read the 18-band window with rasterio.
- Derive a stable class mask from simple spectral/spatial signals:
  - water-like pixels use `open_water`;
  - high vegetation pixels are split across crop/vegetation classes using
    deterministic tile coordinates and NDVI-like ratios;
  - low-vegetation bright pixels use `developed_barren`;
  - remaining pixels fall into `other`.
- Stitch class tiles with `stitch_class_tiles`.
- Compute area summary with `compute_class_area_summary`.
- Preserve the source raster georeferencing in the classified GeoTIFF.

The exact heuristic can be simple. The requirements are determinism,
class-schema stability, raster-size preservation, and clear logs stating that no
real Prithvi checkpoint was used.

### 3. Artifact Writer

Default output directory:

`results/model_hub/prithvi_crop_runs/<input-stem>-<short-hash>/`

Artifacts:

- `classified_crop.tif`: single-band `uint8` classified raster with the same
  CRS, transform, width, and height as the source.
- `crop_preview.png`: optional preview rendered from a downsampled classified
  mask. If PIL is unavailable, omit the preview with a log message instead of
  failing the job.
- `crop_polygons.geojson`: lightweight GIS vector summary. It can contain
  per-class bounding boxes or rasterio-extracted features for classes present in
  the mask. It must be valid GeoJSON.
- `crop_summary.csv`: `class,pixels,fraction` rows for the 13 classes.
- `manifest.json`: ArcGIS-style package manifest containing:
  - `model_id`
  - `input_mode`
  - `source_raster`
  - `input_profile`
  - `output_profile`
  - `validation`
  - `tile_grid`
  - `artifacts`
  - `limitations`

The runtime result should include all artifact paths and the validation report.

### 4. Runtime Dispatch

Update `ae_backend/app/services/model_hub_runtime.py`:

- Keep `cached_demo` dispatch unchanged.
- Add:

```python
if model_id == "prithvi_crop_classification_arcgis_style" and input_mode == "upload_raster_demo":
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo
    return run_prithvi_crop_raster_demo(options=options)
```

Update `ae_backend/app/api/model_hub.py`:

- Execute the crop model synchronously for `upload_raster_demo`.
- Preserve all runtime logs, not just the last log. This fixes the Phase 2
  diagnostic gap where a CSV fallback warning can be dropped from job details.

Update `ae_backend/app/services/model_hub_jobs.py`:

- Add a method or extend `mark_succeeded` so it can append multiple runtime logs.
- Keep the public job schema unchanged: `logs` remains a list of strings.

## API Contract

Request:

```json
{
  "model_id": "prithvi_crop_classification_arcgis_style",
  "input_mode": "upload_raster_demo",
  "options": {
    "raster_path": "D:/tmp/example_prithvi_crop_18band.tif",
    "tile_size": 224,
    "stride": 224,
    "output_formats": ["geotiff", "png", "geojson", "csv", "manifest"]
  }
}
```

Successful response:

```json
{
  "status": "succeeded",
  "result": {
    "task": "crop_classification",
    "model_id": "prithvi_crop_classification_arcgis_style",
    "input_mode": "upload_raster_demo",
    "validation": {
      "band_count": 18,
      "width": 224,
      "height": 224,
      "crs": "EPSG:4326"
    },
    "summary": {
      "class_pixel_counts": {"corn": 1000},
      "class_area_fraction": {"corn": 0.2},
      "dominant_class": "corn",
      "method": "deterministic ArcGIS-style Prithvi crop raster contract demo"
    }
  },
  "artifacts": [
    {"kind": "geotiff", "path": "results/model_hub/prithvi_crop_runs/example/classified_crop.tif"},
    {"kind": "manifest", "path": "results/model_hub/prithvi_crop_runs/example/manifest.json"}
  ],
  "logs": [
    "validated 18-band Prithvi crop raster",
    "ran deterministic tiled crop classification contract demo"
  ]
}
```

Failure examples:

- Missing `options.raster_path`: failed job with `raster_path is required`.
- Wrong band count: failed job with `Prithvi crop raster requires 18 bands`.
- Missing CRS: failed job with `Prithvi crop raster requires georeferencing`.

## Frontend Boundary

Update `ae_frontend/index.html` only inside the Model Hub tab.

Desired behavior:

- Crop package card displays available runtime modes from
  `package_profile.runtime_modes`.
- The quick "run demo" path still uses the default mode from
  `input_spec.default_demo_input_mode`.
- If the selected model supports `upload_raster_demo`, show a compact local path
  input and a secondary action to run raster demo.
- The path input should not appear for unrelated models.
- The recent job JSON remains the authoritative debug view for artifacts and
  validation metadata.

No map viewer or raster preview panel is required in Phase 3. The preview PNG
and classified GeoTIFF are exposed as artifacts.

## Testing Strategy

Use TDD for implementation.

### Unit Tests

Add `tests/test_model_hub_crop_raster.py`.

Required tests:

1. `test_validate_prithvi_crop_raster_accepts_18_band_geotiff`
   - Create a small 18-band GeoTIFF with rasterio.
   - Assert band count, shape, CRS, transform, and band order are returned.

2. `test_validate_prithvi_crop_raster_rejects_wrong_band_count`
   - Create a 6-band GeoTIFF.
   - Assert `ModelHubRuntimeError` mentions 18 bands.

3. `test_run_prithvi_crop_raster_demo_writes_gis_artifacts`
   - Create a 18-band GeoTIFF.
   - Run `run_prithvi_crop_raster_demo` with `output_dir=tmp_path`.
   - Assert result task, dominant class, classified GeoTIFF, CSV, GeoJSON, and
     manifest artifacts.
   - Reopen the classified GeoTIFF and assert shape and CRS match the source.

4. `test_run_prithvi_crop_raster_demo_logs_validation_and_contract_runtime`
   - Assert logs include validation and deterministic-runtime messages.

### API Tests

Extend `tests/test_model_hub_api.py`.

Required tests:

- Posting `upload_raster_demo` for the crop model with a valid test GeoTIFF
  returns `status=succeeded`, `validation.band_count == 18`, and artifact kinds
  include `geotiff`, `csv`, `geojson`, and `manifest`.
- Posting a wrong-band-count raster returns `status=failed` and a clear error.
- A runtime returning multiple logs stores all logs on the job.

### Registry Tests

Extend `tests/test_model_hub_registry.py`.

Required tests:

- Committed crop model registry entry includes `upload_raster_demo`.
- The crop class schema contains the official 13 classes, including `corn`,
  `soybeans`, `open_water`, and `sorghum`.
- `package_profile.input_profile.band_order` has 18 entries.

### Frontend Tests

Extend `tests/test_model_hub_frontend_entry.py`.

Required tests:

- Frontend references `upload_raster_demo`.
- Frontend keeps metadata-driven runtime-mode logic.
- Frontend only exposes the raster path input for models that support
  `upload_raster_demo`.

### Regression Tests

Run after implementation:

```powershell
python -m pytest tests/test_model_hub_registry.py tests/test_model_hub_crop.py tests/test_model_hub_crop_raster.py tests/test_model_hub_jobs.py tests/test_model_hub_api.py tests/test_model_hub_change.py tests/test_raster_pipeline.py tests/test_model_hub_frontend_entry.py tests/test_inference_api.py tests/test_lulc_registry.py -q
python -m pytest tests/test_paper12_public_dataset_results.py tests/test_paper12_colab_notebooks.py -q
git diff --check
```

## Acceptance Criteria

Phase 3 is complete when:

- The crop Model Hub entry advertises both `cached_demo` and
  `upload_raster_demo`.
- A valid local 18-band GeoTIFF can run through
  `POST /api/ae/model-hub/jobs`.
- Invalid rasters fail with clear job errors.
- The runtime writes a classified GeoTIFF, CSV summary, GeoJSON summary, and
  manifest JSON under `results/model_hub/prithvi_crop_runs`.
- The job result includes validation metadata, class-area summary, dominant
  class, model package metadata, artifacts, and all runtime logs.
- Existing cached crop demo, change demo, and LULC demo behavior remains intact.
- Focused tests pass.
- Paper 12 regression tests pass.
- `git diff --check` is clean.

## Risks And Mitigations

- Risk: deterministic raster classification could be mistaken for real Prithvi
  inference.
  Mitigation: every result and manifest must state that no real checkpoint was
  used.

- Risk: accepting local `raster_path` is not suitable for a public server.
  Mitigation: describe it as a trusted local demo API. Multipart upload and
  storage isolation are a later phase.

- Risk: class schema migration breaks old Phase 2 expectations.
  Mitigation: update tests and metadata together. This is an intentional
  alignment with the official 13-class model card.

- Risk: writing preview PNGs adds optional dependencies.
  Mitigation: prefer existing PIL if available, but do not fail the job if only
  the core GeoTIFF/CSV/GeoJSON/manifest artifacts are written.

- Risk: large rasters can be slow.
  Mitigation: process by tile windows and allow `tile_size`/`stride` overrides in
  options. No full-raster 18-band load is required.

## Later Phases

- Add multipart upload with managed storage and path isolation.
- Add real Prithvi crop checkpoint loading with TerraTorch or a local inference
  wrapper.
- Add GPU capability checks and queue-based asynchronous execution.
- Add `.dlpk` export or stricter ArcGIS interoperability packaging.
- Add a map/preview panel for classified crop rasters in the frontend.
- Add regional validation datasets and accuracy reporting.

## References

- ArcGIS AI Models, "Introduction to the model", Prithvi - Crop Classification:
  `https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-prithvi-crop-classification.htm`
- IBM/NASA HuggingFace model card:
  `https://huggingface.co/ibm-nasa-geospatial/prithvi-100m-multi-temporal-crop-classification`
- Phase 2 design:
  `docs/superpowers/specs/2026-06-27-prithvi-crop-model-hub-phase2-design.md`
