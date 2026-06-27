# Prithvi Crop Raster Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ArcGIS-style Prithvi crop raster execution contract that validates local 18-band GeoTIFF inputs and writes deterministic GIS artifacts through Model Hub.

**Architecture:** Keep the current Model Hub shape and add a new crop-specific raster service under `ae_backend/app/services/model_hub_crop_raster.py`. The service validates raster metadata, runs deterministic tiled classification over raster windows, writes artifacts under `results/model_hub/prithvi_crop_runs`, and is dispatched by the existing job API. Frontend changes stay inside the Model Hub tab.

**Tech Stack:** Python 3, FastAPI, rasterio, numpy, Pillow, Vue 3 single-file HTML, pytest.

---

## File Structure

- Modify `ae_backend/app/data/model_hub_models.json`
  - Advertise `upload_raster_demo`, official 13-class schema, 18-band order, and raster artifact profile.
- Modify `ae_backend/app/services/model_hub_jobs.py`
  - Preserve multiple runtime logs in succeeded jobs while keeping existing `log` compatibility.
- Create `ae_backend/app/services/model_hub_crop_raster.py`
  - Validate local 18-band GeoTIFFs and run deterministic tiled crop classification.
- Modify `ae_backend/app/services/model_hub_runtime.py`
  - Dispatch `prithvi_crop_classification_arcgis_style + upload_raster_demo`.
- Modify `ae_backend/app/api/model_hub.py`
  - Execute the new runtime synchronously and store all runtime logs.
- Modify `ae_frontend/index.html`
  - Add Model Hub crop raster path control and action, driven by `package_profile.runtime_modes`.
- Modify `tests/test_model_hub_registry.py`
  - Verify official class schema, runtime modes, and band order.
- Modify `ae_backend/app/services/model_hub_crop.py`
  - Keep cached demo behavior but align class names with the official 13-class crop schema.
- Modify `tests/test_model_hub_crop.py`
  - Update cached demo expectations from Phase 2 local labels to official class names.
- Modify `tests/test_model_hub_jobs.py`
  - Verify multiple runtime logs are preserved.
- Create `tests/test_model_hub_crop_raster.py`
  - Cover validation, failure, artifact writing, and logs.
- Modify `tests/test_model_hub_api.py`
  - Cover valid and invalid `upload_raster_demo` jobs and API log preservation.
- Modify `tests/test_model_hub_frontend_entry.py`
  - Verify frontend raster-demo controls are metadata driven.

---

### Task 1: Registry Contract For Prithvi Crop Raster Mode

**Files:**
- Modify: `tests/test_model_hub_registry.py`
- Modify: `ae_backend/app/data/model_hub_models.json`

- [ ] **Step 1: Write failing registry tests**

In `tests/test_model_hub_registry.py`, update `test_committed_model_hub_registry_loads_prithvi_crop_package` to assert the official schema and 18-band contract:

```python
def test_committed_model_hub_registry_loads_prithvi_crop_package():
    registry = load_model_registry(REGISTRY_DATA_PATH)
    crop = registry.get_model("prithvi_crop_classification_arcgis_style").to_dict()

    assert crop["task_type"] == "crop_classification"
    assert crop["status"] == "demo_only"
    assert crop["input_spec"]["default_demo_input_mode"] == "cached_demo"
    assert crop["input_spec"]["supported_job_input_modes"] == [
        "cached_demo",
        "upload_raster_demo",
    ]
    assert crop["class_schema"] == [
        "natural_vegetation",
        "forest",
        "corn",
        "soybeans",
        "wetlands",
        "developed_barren",
        "open_water",
        "winter_wheat",
        "alfalfa",
        "fallow_idle_cropland",
        "cotton",
        "sorghum",
        "other",
    ]
    assert crop["package_profile"]["family"] == "prithvi_crop_classification"
    assert crop["package_profile"]["runtime_modes"] == [
        "cached_demo",
        "upload_raster_demo",
    ]
    assert len(crop["package_profile"]["input_profile"]["band_order"]) == 18
    assert crop["package_profile"]["input_profile"]["band_order"][:6] == [
        "t1_blue",
        "t1_green",
        "t1_red",
        "t1_narrow_nir",
        "t1_swir1",
        "t1_swir2",
    ]
    assert "classified_raster_geotiff" in crop["package_profile"]["output_profile"]["artifacts"]
    assert crop["package_profile"]["applicability"]["readiness"] == "demo_contract_only"
```

- [ ] **Step 2: Run registry test and verify RED**

Run:

```powershell
python -m pytest tests/test_model_hub_registry.py::test_committed_model_hub_registry_loads_prithvi_crop_package -q
```

Expected: FAIL because `supported_job_input_modes`, `upload_raster_demo`, official classes, and `band_order` are not present yet.

- [ ] **Step 3: Update committed crop registry entry**

In `ae_backend/app/data/model_hub_models.json`, edit only the `prithvi_crop_classification_arcgis_style` record:

```json
"input_spec": {
  "modalities": ["HLS_multitemporal"],
  "bands": 18,
  "patch_size": "224 recommended; variable raster accepted for tiled demo",
  "default_demo_input_mode": "cached_demo",
  "supported_job_input_modes": ["cached_demo", "upload_raster_demo"],
  "normalization": "model_package_defined"
},
"output_spec": {
  "type": "categorical_crop_raster",
  "classes": 13,
  "formats": ["geotiff", "png", "geojson", "csv", "manifest"]
},
"class_schema": [
  "natural_vegetation",
  "forest",
  "corn",
  "soybeans",
  "wetlands",
  "developed_barren",
  "open_water",
  "winter_wheat",
  "alfalfa",
  "fallow_idle_cropland",
  "cotton",
  "sorghum",
  "other"
],
"supported_sensors": ["HLS 3-timestep 18-band composite", "future Sentinel-2/HLS composite"],
"package_profile": {
  "package_type": "arcgis_style_pretrained_imagery_model",
  "family": "prithvi_crop_classification",
  "runtime_modes": ["cached_demo", "upload_raster_demo"],
  "input_profile": {
    "raster_profile": "18_band_hls_multitemporal_composite",
    "requires_georeferencing": true,
    "requires_crop_season_composite": true,
    "band_order": [
      "t1_blue",
      "t1_green",
      "t1_red",
      "t1_narrow_nir",
      "t1_swir1",
      "t1_swir2",
      "t2_blue",
      "t2_green",
      "t2_red",
      "t2_narrow_nir",
      "t2_swir1",
      "t2_swir2",
      "t3_blue",
      "t3_green",
      "t3_red",
      "t3_narrow_nir",
      "t3_swir1",
      "t3_swir2"
    ],
    "notes": "Phase 3 validates local GeoTIFF inputs and runs deterministic contract inference only."
  },
  "output_profile": {
    "primary_output": "categorical crop raster",
    "artifacts": [
      "classified_raster_geotiff",
      "preview_png",
      "crop_polygons_geojson",
      "area_summary_csv",
      "arcgis_style_manifest_json"
    ],
    "class_count": 13
  },
  "applicability": {
    "readiness": "demo_contract_only",
    "region": "real ArcGIS/Prithvi model expected to work well in the United States; local runtime is not accuracy validated",
    "limitations": [
      "No crop checkpoint is wired in this phase.",
      "No ArcGIS .dlpk compatibility is claimed.",
      "Raster demo results are deterministic product-contract artifacts."
    ]
  },
  "model_card": {
    "summary": "ArcGIS-style Prithvi crop classification package contract for 18-band raster demos.",
    "usage": "Run cached_demo for a quick schema check or upload_raster_demo with a local 18-band GeoTIFF path.",
    "next_step": "Attach a validated Prithvi crop head and managed upload pipeline."
  }
}
```

Keep unchanged fields such as `model_id`, `display_name`, `task_type`, `backbone`, `status`, and `license`.

- [ ] **Step 4: Run registry tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_hub_registry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit registry contract**

Run:

```powershell
git add tests/test_model_hub_registry.py ae_backend/app/data/model_hub_models.json
git commit -m "feat: advertise prithvi crop raster runtime"
```

---

### Task 2: Align Cached Crop Demo Class Schema

**Files:**
- Modify: `tests/test_model_hub_crop.py`
- Modify: `ae_backend/app/services/model_hub_crop.py`

- [ ] **Step 1: Update cached-demo tests for official class names**

In `tests/test_model_hub_crop.py`, change the fixture CSV rows and assertions from Phase 2 local labels to official Prithvi crop labels:

```python
(crop_dir / "crop_summary.csv").write_text(
    "class,pixels,fraction\ncorn,6400,0.64\n",
    encoding="utf-8",
)

assert result["result"]["summary"]["dominant_class"] == "corn"
assert result["result"]["summary"]["class_pixel_counts"]["corn"] == 6400
assert result["result"]["summary"]["class_area_fraction"]["corn"] == 0.64
```

For fallback tests, assert the deterministic default summary uses `corn` as the dominant class:

```python
assert result["result"]["summary"]["dominant_class"] == "corn"
assert result["result"]["summary"]["class_pixel_counts"]["corn"] == 6400
assert result["result"]["summary"]["class_area_fraction"]["corn"] == 0.434783
```

- [ ] **Step 2: Run cached crop tests and verify RED**

Run:

```powershell
python -m pytest tests/test_model_hub_crop.py -q
```

Expected: FAIL because `model_hub_crop.py` still returns the Phase 2 local labels such as `maize` and `rice`.

- [ ] **Step 3: Update cached crop demo constants**

In `ae_backend/app/services/model_hub_crop.py`, replace `CROP_CLASSES` and `DEMO_PIXEL_COUNTS` with:

```python
CROP_CLASSES = [
    "natural_vegetation",
    "forest",
    "corn",
    "soybeans",
    "wetlands",
    "developed_barren",
    "open_water",
    "winter_wheat",
    "alfalfa",
    "fallow_idle_cropland",
    "cotton",
    "sorghum",
    "other",
]
DEMO_PIXEL_COUNTS = {
    "natural_vegetation": 1200,
    "forest": 900,
    "corn": 6400,
    "soybeans": 1800,
    "wetlands": 700,
    "developed_barren": 740,
    "open_water": 240,
    "winter_wheat": 900,
    "alfalfa": 420,
    "fallow_idle_cropland": 500,
    "cotton": 300,
    "sorghum": 260,
    "other": 360,
}
```

Keep `summarize_cached_crop_demo` behavior unchanged: it still reads an optional CSV, falls back on malformed data, returns the same artifact kinds, and uses `cached_demo`.

- [ ] **Step 4: Run cached crop tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_hub_crop.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit cached demo schema alignment**

Run:

```powershell
git add tests/test_model_hub_crop.py ae_backend/app/services/model_hub_crop.py
git commit -m "feat: align crop demo class schema"
```

---
### Task 3: Preserve Multiple Model Hub Runtime Logs

**Files:**
- Modify: `tests/test_model_hub_jobs.py`
- Modify: `ae_backend/app/services/model_hub_jobs.py`

- [ ] **Step 1: Write failing job-store log test**

Append this test to `tests/test_model_hub_jobs.py`:

```python
def test_job_store_marks_success_with_multiple_runtime_logs():
    from app.services.model_hub_jobs import ModelHubJobStore

    store = ModelHubJobStore()
    job = store.create_job("prithvi_crop_classification_arcgis_style", "upload_raster_demo", {})
    store.mark_running(job["job_id"], log="job accepted")
    store.mark_succeeded(
        job["job_id"],
        result={"summary": {"dominant_class": "corn"}},
        artifacts=[],
        logs=[
            "validated 18-band Prithvi crop raster",
            "ran deterministic tiled crop classification contract demo",
        ],
    )

    loaded = store.get_job(job["job_id"])
    assert loaded["status"] == "succeeded"
    assert loaded["logs"] == [
        "job accepted",
        "validated 18-band Prithvi crop raster",
        "ran deterministic tiled crop classification contract demo",
    ]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest tests/test_model_hub_jobs.py::test_job_store_marks_success_with_multiple_runtime_logs -q
```

Expected: FAIL with `TypeError` because `mark_succeeded` does not accept `logs`.

- [ ] **Step 3: Extend `mark_succeeded`**

In `ae_backend/app/services/model_hub_jobs.py`, change the method signature and log append block:

```python
    def mark_succeeded(
        self,
        job_id: str,
        result: dict,
        artifacts: list[dict],
        log: str | None = None,
        logs: list[str] | None = None,
    ) -> None:
        job = self._jobs[job_id]
        job["status"] = "succeeded"
        job["result"] = deepcopy(result)
        job["artifacts"] = deepcopy(artifacts)
        job["updated_at"] = _utc_now()
        if log:
            job["logs"].append(log)
        if logs:
            job["logs"].extend(str(item) for item in logs)
```

- [ ] **Step 4: Run job-store tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_hub_jobs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit log preservation**

Run:

```powershell
git add tests/test_model_hub_jobs.py ae_backend/app/services/model_hub_jobs.py
git commit -m "fix: preserve model hub runtime logs"
```

---

### Task 4: Prithvi Crop Raster Validation Service

**Files:**
- Create: `tests/test_model_hub_crop_raster.py`
- Create: `ae_backend/app/services/model_hub_crop_raster.py`

- [ ] **Step 1: Write failing validation tests**

Create `tests/test_model_hub_crop_raster.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def _write_test_geotiff(path: Path, *, bands: int = 18, width: int = 8, height: int = 6) -> Path:
    transform = from_origin(100.0, 40.0, 0.01, 0.01)
    data = np.zeros((bands, height, width), dtype=np.float32)
    for band in range(bands):
        data[band] = (band + 1) / max(bands, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


def test_validate_prithvi_crop_raster_accepts_18_band_geotiff(tmp_path: Path):
    from app.services.model_hub_crop_raster import validate_prithvi_crop_raster

    raster_path = _write_test_geotiff(tmp_path / "crop_18band.tif", bands=18)

    validation = validate_prithvi_crop_raster(raster_path)

    assert validation["band_count"] == 18
    assert validation["width"] == 8
    assert validation["height"] == 6
    assert validation["crs"] == "EPSG:4326"
    assert validation["dtype"] == "float32"
    assert len(validation["band_order"]) == 18
    assert validation["band_order"][:6] == [
        "t1_blue",
        "t1_green",
        "t1_red",
        "t1_narrow_nir",
        "t1_swir1",
        "t1_swir2",
    ]


def test_validate_prithvi_crop_raster_rejects_wrong_band_count(tmp_path: Path):
    from app.services.model_hub_runtime import ModelHubRuntimeError
    from app.services.model_hub_crop_raster import validate_prithvi_crop_raster

    raster_path = _write_test_geotiff(tmp_path / "crop_6band.tif", bands=6)

    with pytest.raises(ModelHubRuntimeError, match="requires 18 bands"):
        validate_prithvi_crop_raster(raster_path)
```

- [ ] **Step 2: Run validation tests and verify RED**

Run:

```powershell
python -m pytest tests/test_model_hub_crop_raster.py -q
```

Expected: FAIL because `app.services.model_hub_crop_raster` does not exist.

- [ ] **Step 3: Implement validator skeleton**

Create `ae_backend/app/services/model_hub_crop_raster.py` with this initial content:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
from rasterio.errors import RasterioIOError

from app.services.model_hub_runtime import ModelHubRuntimeError


CROP_RASTER_MODEL_ID = "prithvi_crop_classification_arcgis_style"
CROP_RASTER_CLASSES = [
    "natural_vegetation",
    "forest",
    "corn",
    "soybeans",
    "wetlands",
    "developed_barren",
    "open_water",
    "winter_wheat",
    "alfalfa",
    "fallow_idle_cropland",
    "cotton",
    "sorghum",
    "other",
]
CROP_RASTER_BAND_ORDER = [
    "t1_blue",
    "t1_green",
    "t1_red",
    "t1_narrow_nir",
    "t1_swir1",
    "t1_swir2",
    "t2_blue",
    "t2_green",
    "t2_red",
    "t2_narrow_nir",
    "t2_swir1",
    "t2_swir2",
    "t3_blue",
    "t3_green",
    "t3_red",
    "t3_narrow_nir",
    "t3_swir1",
    "t3_swir2",
]


def _as_jsonable_bounds(bounds: Any) -> list[float]:
    return [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)]


def validate_prithvi_crop_raster(raster_path: str | Path) -> dict:
    path = Path(raster_path)
    if not path.exists():
        raise ModelHubRuntimeError(f"Prithvi crop raster does not exist: {path}")
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ModelHubRuntimeError("Prithvi crop raster must be a GeoTIFF .tif or .tiff file")

    try:
        with rasterio.open(path) as src:
            if src.count != 18:
                raise ModelHubRuntimeError(
                    f"Prithvi crop raster requires 18 bands, got {src.count}"
                )
            if src.width <= 0 or src.height <= 0:
                raise ModelHubRuntimeError("Prithvi crop raster width and height must be positive")
            if src.crs is None:
                raise ModelHubRuntimeError("Prithvi crop raster requires georeferencing CRS")
            if src.transform is None:
                raise ModelHubRuntimeError("Prithvi crop raster requires georeferencing transform")
            return {
                "path": str(path),
                "band_count": int(src.count),
                "width": int(src.width),
                "height": int(src.height),
                "crs": src.crs.to_string(),
                "transform": [float(value) for value in src.transform.to_gdal()],
                "bounds": _as_jsonable_bounds(src.bounds),
                "dtype": str(src.dtypes[0]),
                "nodata": [None if value is None else float(value) for value in src.nodatavals],
                "band_order": list(CROP_RASTER_BAND_ORDER),
            }
    except RasterioIOError as exc:
        raise ModelHubRuntimeError(f"Could not open Prithvi crop raster: {path}") from exc
```

- [ ] **Step 4: Run validation tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_hub_crop_raster.py -q
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit validator**

Run:

```powershell
git add tests/test_model_hub_crop_raster.py ae_backend/app/services/model_hub_crop_raster.py
git commit -m "feat: validate prithvi crop rasters"
```

---

### Task 5: Deterministic Raster Demo Artifacts

**Files:**
- Modify: `tests/test_model_hub_crop_raster.py`
- Modify: `ae_backend/app/services/model_hub_crop_raster.py`

- [ ] **Step 1: Write failing artifact runtime tests**

Append to `tests/test_model_hub_crop_raster.py`:

```python
def test_run_prithvi_crop_raster_demo_writes_gis_artifacts(tmp_path: Path):
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

    raster_path = _write_test_geotiff(tmp_path / "crop_18band.tif", bands=18, width=12, height=10)
    output_dir = tmp_path / "outputs"

    result = run_prithvi_crop_raster_demo(
        options={
            "raster_path": str(raster_path),
            "output_dir": str(output_dir),
            "tile_size": 6,
            "stride": 6,
        }
    )

    assert result["result"]["task"] == "crop_classification"
    assert result["result"]["input_mode"] == "upload_raster_demo"
    assert result["result"]["validation"]["band_count"] == 18
    assert result["result"]["summary"]["dominant_class"] in result["result"]["model_package"]["class_schema"]
    artifact_by_kind = {artifact["kind"]: Path(artifact["path"]) for artifact in result["artifacts"]}
    assert {"geotiff", "csv", "geojson", "manifest"}.issubset(artifact_by_kind)
    assert artifact_by_kind["geotiff"].exists()
    assert artifact_by_kind["csv"].exists()
    assert artifact_by_kind["geojson"].exists()
    assert artifact_by_kind["manifest"].exists()

    with rasterio.open(artifact_by_kind["geotiff"]) as classified:
        assert classified.count == 1
        assert classified.width == 12
        assert classified.height == 10
        assert classified.crs.to_string() == "EPSG:4326"


def test_run_prithvi_crop_raster_demo_logs_validation_and_contract_runtime(tmp_path: Path):
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

    raster_path = _write_test_geotiff(tmp_path / "crop_18band.tif", bands=18)

    result = run_prithvi_crop_raster_demo(
        options={"raster_path": str(raster_path), "output_dir": str(tmp_path / "outputs")}
    )

    assert any("validated 18-band Prithvi crop raster" in log for log in result["logs"])
    assert any("deterministic tiled crop classification" in log for log in result["logs"])
    assert any("no real Prithvi checkpoint" in log for log in result["logs"])
```

- [ ] **Step 2: Run new tests and verify RED**

Run:

```powershell
python -m pytest tests/test_model_hub_crop_raster.py -q
```

Expected: FAIL because `run_prithvi_crop_raster_demo` does not exist.

- [ ] **Step 3: Implement deterministic runtime and artifact writers**

Extend `ae_backend/app/services/model_hub_crop_raster.py` with these imports:

```python
import csv
import hashlib
import json

import numpy as np
from PIL import Image
from rasterio import features

from app.core.config import PROJECT_ROOT
from app.services.raster_pipeline import (
    compute_class_area_summary,
    make_tile_grid,
    stitch_class_tiles,
)
```

Add helper functions:

```python
def _default_output_dir(raster_path: Path) -> Path:
    digest = hashlib.sha1(str(raster_path.resolve()).encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raster_path.stem)
    return Path(PROJECT_ROOT) / "results" / "model_hub" / "prithvi_crop_runs" / f"{safe_stem}-{digest}"


def _class_id(name: str) -> int:
    return CROP_RASTER_CLASSES.index(name)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)


def _classify_tile(tile: np.ndarray, *, tile_index: int) -> np.ndarray:
    tile = np.asarray(tile, dtype=np.float32)
    blue = tile[0]
    red = tile[2]
    nir = tile[3]
    swir1 = tile[4]
    green = tile[1]
    ndvi = _safe_ratio(nir - red, nir + red)
    ndwi = _safe_ratio(green - nir, green + nir)
    brightness = np.nanmean(tile[:6], axis=0)

    mask = np.full(red.shape, _class_id("other"), dtype=np.uint8)
    mask[ndwi > 0.15] = _class_id("open_water")
    mask[(ndvi < 0.05) & (brightness > 0.35)] = _class_id("developed_barren")
    vegetation = ndvi > 0.2
    crop_classes = [
        "corn",
        "soybeans",
        "winter_wheat",
        "alfalfa",
        "cotton",
        "sorghum",
        "natural_vegetation",
        "forest",
        "fallow_idle_cropland",
        "wetlands",
    ]
    yy, xx = np.indices(mask.shape)
    class_offsets = (xx + yy + tile_index) % len(crop_classes)
    for offset, class_name in enumerate(crop_classes):
        mask[vegetation & (class_offsets == offset)] = _class_id(class_name)
    return mask


def _write_classified_raster(path: Path, mask: np.ndarray, src_profile: dict) -> None:
    profile = dict(src_profile)
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def _write_summary_csv(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=["class", "pixels", "fraction"])
        writer.writeheader()
        for class_name in CROP_RASTER_CLASSES:
            writer.writerow(
                {
                    "class": class_name,
                    "pixels": summary["class_pixel_counts"][class_name],
                    "fraction": summary["class_area_fraction"][class_name],
                }
            )


def _write_preview_png(path: Path, mask: np.ndarray) -> bool:
    palette = np.array(
        [
            [70, 150, 70],
            [20, 95, 45],
            [238, 189, 34],
            [76, 175, 80],
            [61, 125, 94],
            [160, 160, 145],
            [53, 130, 190],
            [216, 188, 84],
            [125, 180, 90],
            [190, 165, 90],
            [220, 125, 80],
            [190, 95, 65],
            [130, 130, 130],
        ],
        dtype=np.uint8,
    )
    rgb = palette[np.clip(mask.astype(np.int64), 0, len(palette) - 1)]
    Image.fromarray(rgb, mode="RGB").save(path)
    return True


def _write_geojson(path: Path, mask: np.ndarray, transform) -> None:
    features_out = []
    for geom, value in features.shapes(mask.astype(np.uint8), transform=transform):
        class_id = int(value)
        if 0 <= class_id < len(CROP_RASTER_CLASSES):
            features_out.append(
                {
                    "type": "Feature",
                    "properties": {
                        "class_id": class_id,
                        "class_name": CROP_RASTER_CLASSES[class_id],
                    },
                    "geometry": geom,
                }
            )
    feature_collection = {"type": "FeatureCollection", "features": features_out}
    path.write_text(json.dumps(feature_collection), encoding="utf-8")
```

Add `run_prithvi_crop_raster_demo`:

```python
def run_prithvi_crop_raster_demo(*, options: dict) -> dict:
    raster_path_value = options.get("raster_path")
    if not raster_path_value:
        raise ModelHubRuntimeError("raster_path is required for upload_raster_demo")

    raster_path = Path(raster_path_value)
    validation = validate_prithvi_crop_raster(raster_path)
    output_dir = Path(options.get("output_dir") or _default_output_dir(raster_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = int(options.get("tile_size") or 224)
    stride = int(options.get("stride") or tile_size)
    logs = [
        f"validated 18-band Prithvi crop raster from {raster_path}",
        "using deterministic tiled crop classification; no real Prithvi checkpoint was loaded",
    ]

    with rasterio.open(raster_path) as src:
        tiles = make_tile_grid(src.width, src.height, tile_size, stride)
        tile_masks = []
        for tile_index, window in enumerate(tiles):
            raster_window = rasterio.windows.Window(
                col_off=window["x0"],
                row_off=window["y0"],
                width=window["x1"] - window["x0"],
                height=window["y1"] - window["y0"],
            )
            tile = src.read(window=raster_window, boundless=False)
            tile_masks.append((window, _classify_tile(tile, tile_index=tile_index)))
        mask = stitch_class_tiles(width=src.width, height=src.height, tiles=tile_masks, fill_value=_class_id("other"))
        src_profile = src.profile
        src_transform = src.transform

    summary = compute_class_area_summary(mask, CROP_RASTER_CLASSES)
    dominant_class = max(summary["class_pixel_counts"], key=summary["class_pixel_counts"].get)

    classified_path = output_dir / "classified_crop.tif"
    csv_path = output_dir / "crop_summary.csv"
    geojson_path = output_dir / "crop_polygons.geojson"
    manifest_path = output_dir / "manifest.json"
    preview_path = output_dir / "crop_preview.png"

    _write_classified_raster(classified_path, mask, src_profile)
    _write_summary_csv(csv_path, summary)
    _write_geojson(geojson_path, mask, src_transform)
    artifacts = [
        {"kind": "geotiff", "path": str(classified_path)},
        {"kind": "csv", "path": str(csv_path)},
        {"kind": "geojson", "path": str(geojson_path)},
        {"kind": "manifest", "path": str(manifest_path)},
    ]
    try:
        _write_preview_png(preview_path, mask)
        artifacts.insert(1, {"kind": "png", "path": str(preview_path)})
    except Exception as exc:
        logs.append(f"preview png skipped: {exc}")

    manifest = {
        "model_id": CROP_RASTER_MODEL_ID,
        "input_mode": "upload_raster_demo",
        "source_raster": str(raster_path),
        "input_profile": {"band_order": list(CROP_RASTER_BAND_ORDER), "band_count": 18},
        "output_profile": {"class_schema": list(CROP_RASTER_CLASSES), "artifacts": artifacts},
        "validation": validation,
        "tile_grid": {"tile_size": tile_size, "stride": stride, "tile_count": len(tiles)},
        "limitations": ["deterministic contract demo", "no real Prithvi checkpoint was loaded"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logs.append(f"ran deterministic tiled crop classification over {len(tiles)} tile(s)")
    logs.append(f"wrote crop raster artifacts to {output_dir}")

    return {
        "result": {
            "task": "crop_classification",
            "model_id": CROP_RASTER_MODEL_ID,
            "input_mode": "upload_raster_demo",
            "validation": validation,
            "summary": {
                **summary,
                "dominant_class": dominant_class,
                "method": "deterministic ArcGIS-style Prithvi crop raster contract demo",
            },
            "model_package": {
                "package_type": "arcgis_style_pretrained_imagery_model",
                "family": "prithvi_crop_classification",
                "runtime_mode": "upload_raster_demo",
                "class_schema": list(CROP_RASTER_CLASSES),
            },
        },
        "artifacts": artifacts,
        "logs": logs,
    }
```

- [ ] **Step 4: Run crop raster tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_hub_crop_raster.py -q
```

Expected: all crop raster service tests PASS.

- [ ] **Step 5: Run existing raster pipeline tests**

Run:

```powershell
python -m pytest tests/test_raster_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit raster runtime**

Run:

```powershell
git add tests/test_model_hub_crop_raster.py ae_backend/app/services/model_hub_crop_raster.py
git commit -m "feat: write prithvi crop raster demo artifacts"
```

---

### Task 6: Model Hub API Dispatch For Raster Demo

**Files:**
- Modify: `tests/test_model_hub_api.py`
- Modify: `ae_backend/app/services/model_hub_runtime.py`
- Modify: `ae_backend/app/api/model_hub.py`

- [ ] **Step 1: Add API tests**

In `tests/test_model_hub_api.py`, add imports near the top:

```python
import numpy as np
import rasterio
from rasterio.transform import from_origin
```

Add helper:

```python
def _write_api_test_geotiff(path: Path, *, bands: int = 18) -> Path:
    data = np.zeros((bands, 8, 8), dtype=np.float32)
    for band in range(bands):
        data[band] = (band + 1) / max(bands, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(100.0, 40.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)
    return path
```

Append tests:

```python
def test_model_hub_runs_prithvi_crop_upload_raster_demo_job(tmp_path: Path):
    from app.main import app

    raster_path = _write_api_test_geotiff(tmp_path / "crop_18band.tif", bands=18)
    output_dir = tmp_path / "outputs"
    client = TestClient(app)

    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "upload_raster_demo",
            "options": {
                "raster_path": str(raster_path),
                "output_dir": str(output_dir),
                "tile_size": 4,
                "stride": 4,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["input_mode"] == "upload_raster_demo"
    assert body["result"]["validation"]["band_count"] == 18
    assert {"geotiff", "csv", "geojson", "manifest"}.issubset(
        {artifact["kind"] for artifact in body["artifacts"]}
    )
    assert any("validated 18-band Prithvi crop raster" in log for log in body["logs"])
    assert any("deterministic tiled crop classification" in log for log in body["logs"])


def test_model_hub_fails_prithvi_crop_upload_raster_demo_for_wrong_band_count(tmp_path: Path):
    from app.main import app

    raster_path = _write_api_test_geotiff(tmp_path / "crop_6band.tif", bands=6)
    client = TestClient(app)

    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "upload_raster_demo",
            "options": {"raster_path": str(raster_path)},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "requires 18 bands" in body["error"]


def test_model_hub_api_preserves_all_runtime_logs(monkeypatch):
    from app.main import app
    import app.api.model_hub as model_hub_api

    def fake_run_model_hub_job(*, model_id, input_mode, options):
        assert model_id == "prithvi_crop_classification_arcgis_style"
        assert input_mode == "upload_raster_demo"
        return {
            "result": {"task": "crop_classification", "summary": {"dominant_class": "corn"}},
            "artifacts": [],
            "logs": ["first runtime log", "second runtime log"],
        }

    monkeypatch.setattr(model_hub_api, "run_model_hub_job", fake_run_model_hub_job)
    client = TestClient(app)

    response = client.post(
        "/api/ae/model-hub/jobs",
        json={
            "model_id": "prithvi_crop_classification_arcgis_style",
            "input_mode": "upload_raster_demo",
            "options": {"raster_path": "unused.tif"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["logs"][-2:] == ["first runtime log", "second runtime log"]
```

- [ ] **Step 2: Run API tests and verify RED**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py::test_model_hub_runs_prithvi_crop_upload_raster_demo_job tests/test_model_hub_api.py::test_model_hub_fails_prithvi_crop_upload_raster_demo_for_wrong_band_count tests/test_model_hub_api.py::test_model_hub_api_preserves_all_runtime_logs -q
```

Expected: FAIL because `upload_raster_demo` is not executed synchronously and runtime dispatch is missing.

- [ ] **Step 3: Add runtime dispatch**

In `ae_backend/app/services/model_hub_runtime.py`, add before the final `raise`:

```python
    if model_id == "prithvi_crop_classification_arcgis_style" and input_mode == "upload_raster_demo":
        from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

        return run_prithvi_crop_raster_demo(options=options)
```

- [ ] **Step 4: Execute supported runtime modes in API**

In `ae_backend/app/api/model_hub.py`, replace the `should_execute_now` expression with a registry-driven helper:

```python
def _runtime_modes_for_model(model: dict) -> set[str]:
    runtime_modes = set(model.get("package_profile", {}).get("runtime_modes", []))
    default_mode = model.get("input_spec", {}).get("default_demo_input_mode")
    if default_mode:
        runtime_modes.add(default_mode)
    if model.get("model_id") == "lulc_6class_prithvi_houlsby":
        runtime_modes.add("demo_patch")
    return runtime_modes
```

Then in `create_job`, store the entry:

```python
    try:
        model_entry = get_model_registry().get_model(request.model_id).to_dict()
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id: {request.model_id}",
        ) from exc
```

Use:

```python
    should_execute_now = request.input_mode in _runtime_modes_for_model(model_entry)
```

When marking success, pass all logs:

```python
        JOB_STORE.mark_succeeded(
            job["job_id"],
            result=runtime_result["result"],
            artifacts=runtime_result["artifacts"],
            logs=runtime_result.get("logs", []),
        )
```

- [ ] **Step 5: Run API tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_hub_api.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit API dispatch**

Run:

```powershell
git add tests/test_model_hub_api.py ae_backend/app/services/model_hub_runtime.py ae_backend/app/api/model_hub.py
git commit -m "feat: run prithvi crop raster model hub jobs"
```

---

### Task 7: Frontend Raster Demo Controls

**Files:**
- Modify: `tests/test_model_hub_frontend_entry.py`
- Modify: `ae_frontend/index.html`

- [ ] **Step 1: Write failing frontend tests**

Update `tests/test_model_hub_frontend_entry.py`:

```python
def test_frontend_uses_model_hub_runtime_modes_for_demo_jobs():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "getModelHubDemoInputMode" in html
    assert "package_profile?.runtime_modes" in html
    assert "default_demo_input_mode" in html
    assert "model.task_type === 'change_detection'" in html
    assert "model.status === 'demo_only'" in html
    assert "model.model_id === 'lulc_6class_prithvi_houlsby'" in html
    assert "prithvi_crop_classification_arcgis_style" not in html


def test_frontend_exposes_metadata_driven_crop_raster_demo_controls():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "upload_raster_demo" in html
    assert "supportsModelHubRasterDemo" in html
    assert "modelHubRasterPath" in html
    assert "runModelHubRasterDemo" in html
    assert "supportsModelHubRasterDemo(model)" in html
```

- [ ] **Step 2: Run frontend tests and verify RED**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: FAIL because raster demo controls do not exist.

- [ ] **Step 3: Add frontend state and helpers**

In `ae_frontend/index.html`, near existing Model Hub refs:

```javascript
            const modelHubRasterPath = ref('');
```

Near `isModelHubRunnable`:

```javascript
            const supportsModelHubRasterDemo = (model) => {
                const runtimeModes = model?.package_profile?.runtime_modes || [];
                return runtimeModes.includes('upload_raster_demo');
            };
```

Add action near `runModelHubDemo`:

```javascript
            const runModelHubRasterDemo = async (model) => {
                if (!supportsModelHubRasterDemo(model)) {
                    setModelHubStatus('info', `${model.display_name} does not declare raster demo mode.`);
                    return;
                }
                if (!modelHubRasterPath.value.trim()) {
                    setModelHubStatus('error', 'Enter a local 18-band GeoTIFF path first.');
                    return;
                }
                runningModelHubJob.value = true;
                try {
                    const res = await fetch('/api/ae/model-hub/jobs', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_id: model.model_id,
                            input_mode: 'upload_raster_demo',
                            options: {
                                raster_path: modelHubRasterPath.value.trim(),
                                output_formats: ['geotiff', 'png', 'geojson', 'csv', 'manifest'],
                            },
                        }),
                    });
                    if (!res.ok) throw new Error(await res.text());
                    modelHubJob.value = await res.json();
                    setModelHubStatus('info', `Raster job ${modelHubJob.value.job_id} status: ${modelHubJob.value.status}`);
                } catch (e) {
                    console.error('Failed to run model hub raster job', e);
                    setModelHubStatus('error', 'Raster Model Hub job failed. Check backend logs.');
                } finally {
                    runningModelHubJob.value = false;
                }
            };
```

Return the new symbols from `setup()`:

```javascript
                modelHubRasterPath, supportsModelHubRasterDemo, runModelHubRasterDemo,
```

- [ ] **Step 4: Add compact card controls**

Inside the Model Hub card, after the existing run-demo button block, add:

```html
                        <div v-if="supportsModelHubRasterDemo(model)" class="grid grid-cols-1 gap-2 pt-2 border-t border-gray-100">
                            <input v-model="modelHubRasterPath" type="text" placeholder="Local 18-band GeoTIFF path, for example D:/tmp/crop_18band.tif" class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-md text-xs focus:ring-primary focus:border-primary outline-none" />
                            <button class="px-3 py-2 bg-primary text-white rounded-md text-sm hover:bg-blue-600 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed" :disabled="runningModelHubJob" @click="runModelHubRasterDemo(model)">
                                {{ runningModelHubJob ? 'Running...' : 'Run Raster Demo' }}
                            </button>
                        </div>
```

- [ ] **Step 5: Run frontend tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_hub_frontend_entry.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit frontend controls**

Run:

```powershell
git add tests/test_model_hub_frontend_entry.py ae_frontend/index.html
git commit -m "feat: expose crop raster demo controls"
```

---

### Task 8: Focused Regression And Integration Cleanup

**Files:**
- Review all modified files.
- No new files unless a bug is found during verification.

- [ ] **Step 1: Run focused Model Hub suite**

Run:

```powershell
python -m pytest tests/test_model_hub_registry.py tests/test_model_hub_crop.py tests/test_model_hub_crop_raster.py tests/test_model_hub_jobs.py tests/test_model_hub_api.py tests/test_model_hub_change.py tests/test_raster_pipeline.py tests/test_model_hub_frontend_entry.py tests/test_inference_api.py tests/test_lulc_registry.py -q
```

Expected: all tests PASS. The known Windows Torch DLL access violation may print after pytest output with exit code 0.

- [ ] **Step 2: Run Paper 12 regression tests**

Run:

```powershell
python -m pytest tests/test_paper12_public_dataset_results.py tests/test_paper12_colab_notebooks.py -q
```

Expected: PASS.

- [ ] **Step 3: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no output, exit code 0.

- [ ] **Step 4: Inspect final diff**

Run:

```powershell
git diff --stat master...HEAD
git status --short --branch
```

Expected:

- Branch is `paper12-prithvi-raster-phase3`.
- Worktree is clean after commits.
- Diff includes only the planned registry, service, API, frontend, test, spec, and plan files.

- [ ] **Step 5: Commit any verification fixes**

If verification required small fixes, commit them:

```powershell
git add <changed-files>
git commit -m "test: verify prithvi crop raster phase 3"
```

Skip this commit if no files changed after Task 6.

## Plan Self-Review Checklist

- Spec coverage:
  - Registry runtime modes and official class schema: Task 1.
  - Cached crop demo schema alignment: Task 2.
  - Multiple log preservation: Task 3 and Task 6.
  - Raster validation: Task 4.
  - Tiled deterministic runtime and GIS artifacts: Task 5.
  - API job support: Task 6.
  - Frontend path control: Task 7.
  - Focused and Paper 12 regressions: Task 8.
- Incomplete-marker scan:
  - No unfinished-marker steps.
  - Every task has exact files, test commands, expected failures, implementation snippets, and commit commands.
- Type consistency:
  - Runtime mode is consistently `upload_raster_demo`.
  - Model id is consistently `prithvi_crop_classification_arcgis_style`.
  - Artifact kinds are consistently `geotiff`, `png`, `csv`, `geojson`, and `manifest`.
  - Official class names match the Phase 3 design spec.
