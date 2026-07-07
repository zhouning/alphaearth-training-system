import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def _write_rgb_geotiff(path: Path) -> Path:
    data = np.zeros((3, 16, 16), dtype=np.uint8)
    data[0] = 120
    data[1] = 90
    data[2] = 60
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(107.0, 41.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)
    return path


def test_lulc_raster_runtime_writes_gis_artifacts(tmp_path: Path, monkeypatch):
    import app.services.model_hub_lulc_raster as service

    raster_path = _write_rgb_geotiff(tmp_path / "rgb.tif")
    output_dir = tmp_path / "out"

    def fake_predict_image(image, model_id, checkpoint_path):
        h, w, _ = image.shape
        mask = np.full((h, w), 2, dtype=np.int64)
        return {
            "model_id": model_id or "fake-lulc",
            "mask": mask.tolist(),
            "classes": [
                "background",
                "built",
                "crops",
                "trees",
                "water",
                "rangeland_bare",
            ],
            "class_pixel_counts": {"crops": int(mask.size)},
            "class_area_fraction": {"crops": 1.0},
            "mask_shape": [h, w],
        }

    monkeypatch.setattr(service, "_predict_rgb_tile", fake_predict_image)

    result = service.run_lulc_raster_inference(
        options={
            "raster_path": str(raster_path),
            "output_dir": str(output_dir),
            "tile_size": 8,
            "stride": 8,
        }
    )

    assert result["result"]["task"] == "lulc_segmentation"
    kinds = {artifact["kind"] for artifact in result["artifacts"]}
    assert {"geotiff", "csv", "geojson", "manifest", "png"}.issubset(kinds)
    with rasterio.open(output_dir / "classified_lulc.tif") as classified:
        assert classified.width == 16
        assert classified.height == 16
        assert classified.crs.to_string() == "EPSG:4326"


def test_lulc_raster_runtime_rejects_wrong_band_count(tmp_path: Path):
    from app.services.model_hub_lulc_raster import validate_lulc_raster
    from app.services.model_hub_runtime import ModelHubRuntimeError

    bad_path = tmp_path / "bad.tif"
    data = np.zeros((4, 8, 8), dtype=np.uint8)
    with rasterio.open(
        bad_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(107.0, 41.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)

    try:
        validate_lulc_raster(bad_path)
    except ModelHubRuntimeError as exc:
        assert "requires 3 bands" in str(exc)
    else:
        raise AssertionError("validate_lulc_raster accepted a non-RGB GeoTIFF")
