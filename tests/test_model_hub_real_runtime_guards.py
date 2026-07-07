import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def _write_geotiff(path: Path, bands: int) -> Path:
    data = np.zeros((bands, 8, 8), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(100.0, 40.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)
    return path


def test_real_crop_runtime_rejects_missing_weights(tmp_path: Path):
    from app.services.model_hub_real_crop import run_real_crop_inference
    from app.services.model_hub_runtime import ModelHubRuntimeError

    raster = _write_geotiff(tmp_path / "crop.tif", bands=18)
    with pytest.raises(ModelHubRuntimeError, match="crop.*weights|Prithvi.*weights"):
        run_real_crop_inference(
            options={"raster_path": str(raster), "weights_dir": str(tmp_path / "missing")}
        )


def test_real_flood_runtime_validates_six_bands(tmp_path: Path):
    from app.services.model_hub_flood import validate_flood_raster

    raster = _write_geotiff(tmp_path / "flood.tif", bands=6)
    validation = validate_flood_raster(raster)
    assert validation["band_count"] == 6
    assert validation["width"] == 8


def test_real_flood_runtime_rejects_wrong_band_count(tmp_path: Path):
    from app.services.model_hub_flood import validate_flood_raster
    from app.services.model_hub_runtime import ModelHubRuntimeError

    raster = _write_geotiff(tmp_path / "flood_bad.tif", bands=3)
    with pytest.raises(ModelHubRuntimeError, match="requires 6 bands"):
        validate_flood_raster(raster)

def test_model_hub_runtime_dispatches_real_crop(monkeypatch):
    from app.services.model_hub_runtime import run_model_hub_job
    import app.services.model_hub_real_crop as crop_runtime

    def fake_run_real_crop_inference(*, options):
        assert options == {"raster_path": "crop.tif"}
        return {"result": {"task": "crop_classification"}, "artifacts": [], "logs": []}

    monkeypatch.setattr(crop_runtime, "run_real_crop_inference", fake_run_real_crop_inference)

    result = run_model_hub_job(
        model_id="prithvi_crop_classification_arcgis_style",
        input_mode="real_raster_inference",
        options={"raster_path": "crop.tif"},
    )

    assert result["result"]["task"] == "crop_classification"


def test_model_hub_runtime_dispatches_real_flood(monkeypatch):
    from app.services.model_hub_runtime import run_model_hub_job
    import app.services.model_hub_flood as flood_runtime

    def fake_run_real_flood_inference(*, options):
        assert options == {"raster_path": "flood.tif"}
        return {"result": {"task": "flood_mapping"}, "artifacts": [], "logs": []}

    monkeypatch.setattr(flood_runtime, "run_real_flood_inference", fake_run_real_flood_inference)

    result = run_model_hub_job(
        model_id="water_flood_prithvi",
        input_mode="real_raster_inference",
        options={"raster_path": "flood.tif"},
    )

    assert result["result"]["task"] == "flood_mapping"

