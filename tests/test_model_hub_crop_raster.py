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


def _write_ungeoreferenced_geotiff(path: Path) -> Path:
    data = np.zeros((18, 6, 8), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=6,
        width=8,
        count=18,
        dtype="float32",
        crs="EPSG:4326",
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


def test_validate_prithvi_crop_raster_rejects_missing_transform(tmp_path: Path):
    from app.services.model_hub_runtime import ModelHubRuntimeError
    from app.services.model_hub_crop_raster import validate_prithvi_crop_raster

    raster_path = _write_ungeoreferenced_geotiff(tmp_path / "crop_no_transform.tif")

    with pytest.raises(ModelHubRuntimeError, match="requires georeferencing transform"):
        validate_prithvi_crop_raster(raster_path)
