import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def _write_test_geotiff(
    path: Path,
    *,
    bands: int = 18,
    width: int = 8,
    height: int = 6,
    crs: str = "EPSG:4326",
    transform=None,
) -> Path:
    transform = transform or from_origin(100.0, 40.0, 0.01, 0.01)
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
        crs=crs,
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

    with rasterio.open(raster_path) as source, rasterio.open(artifact_by_kind["geotiff"]) as classified:
        assert classified.count == 1
        assert classified.width == 12
        assert classified.height == 10
        assert classified.crs.to_string() == "EPSG:4326"
        assert classified.transform == source.transform
        assert classified.bounds == source.bounds

    with artifact_by_kind["csv"].open(encoding="utf-8", newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert len(rows) == len(result["result"]["model_package"]["class_schema"])
    assert {row["class"] for row in rows} == set(result["result"]["model_package"]["class_schema"])

    geojson = json.loads(artifact_by_kind["geojson"].read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection"

    manifest = json.loads(artifact_by_kind["manifest"].read_text(encoding="utf-8"))
    assert manifest["artifacts"] == result["artifacts"]


def test_run_prithvi_crop_raster_demo_logs_validation_and_contract_runtime(tmp_path: Path):
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

    raster_path = _write_test_geotiff(tmp_path / "crop_18band.tif", bands=18)

    result = run_prithvi_crop_raster_demo(
        options={"raster_path": str(raster_path), "output_dir": str(tmp_path / "outputs")}
    )

    assert any("validated 18-band Prithvi crop raster" in log for log in result["logs"])
    assert any("deterministic tiled crop classification" in log for log in result["logs"])
    assert any("no real Prithvi checkpoint" in log for log in result["logs"])


@pytest.mark.parametrize(
    ("option_name", "option_value", "message"),
    [
        ("tile_size", "not-an-int", "tile_size.*positive integer"),
        ("tile_size", 0, "tile_size.*at least 1"),
        ("tile_size", -4, "tile_size.*at least 1"),
        ("stride", "not-an-int", "stride.*positive integer"),
        ("stride", 0, "stride.*at least 1"),
        ("stride", -2, "stride.*at least 1"),
        ("max_pixels", 2_000_001, "max_pixels.*at most"),
        ("max_tiles", 4097, "max_tiles.*at most"),
        ("max_preview_pixels", 1_000_001, "max_preview_pixels.*at most"),
        ("max_geojson_features", 5001, "max_geojson_features.*at most"),
    ],
)
def test_run_prithvi_crop_raster_demo_rejects_invalid_tile_options(
    tmp_path: Path,
    option_name: str,
    option_value: object,
    message: str,
):
    from app.services.model_hub_runtime import ModelHubRuntimeError
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

    raster_path = _write_test_geotiff(tmp_path / "crop_18band.tif", bands=18)
    options = {"raster_path": str(raster_path), "output_dir": str(tmp_path / "outputs")}
    options[option_name] = option_value

    with pytest.raises(ModelHubRuntimeError, match=message):
        run_prithvi_crop_raster_demo(options=options)


def test_run_prithvi_crop_raster_demo_records_geojson_feature_limit(tmp_path: Path):
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

    raster_path = _write_test_geotiff(tmp_path / "crop_18band.tif", bands=18, width=12, height=10)

    result = run_prithvi_crop_raster_demo(
        options={
            "raster_path": str(raster_path),
            "output_dir": str(tmp_path / "outputs"),
            "tile_size": 2,
            "stride": 2,
            "max_geojson_features": 3,
        }
    )

    artifact_by_kind = {artifact["kind"]: Path(artifact["path"]) for artifact in result["artifacts"]}
    geojson = json.loads(artifact_by_kind["geojson"].read_text(encoding="utf-8"))
    manifest = json.loads(artifact_by_kind["manifest"].read_text(encoding="utf-8"))

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) <= 3
    assert manifest["geojson_policy"]["max_features"] == 3
    assert manifest["geojson_policy"]["features_truncated"] is True
    assert any("GeoJSON feature limit" in log for log in result["logs"])

def _first_geojson_coordinate(geometry: dict) -> tuple[float, float]:
    coordinates = geometry["coordinates"]
    while coordinates and isinstance(coordinates[0][0], (list, tuple)):
        coordinates = coordinates[0]
    x, y = coordinates[0]
    return float(x), float(y)


def test_run_prithvi_crop_raster_demo_rejects_resource_limits(tmp_path: Path):
    from app.services.model_hub_runtime import ModelHubRuntimeError
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

    raster_path = _write_test_geotiff(tmp_path / "crop_18band.tif", width=8, height=6)

    with pytest.raises(ModelHubRuntimeError, match="max_pixels"):
        run_prithvi_crop_raster_demo(
            options={
                "raster_path": str(raster_path),
                "output_dir": str(tmp_path / "outputs_pixels"),
                "max_pixels": 10,
            }
        )

    with pytest.raises(ModelHubRuntimeError, match="max_tiles"):
        run_prithvi_crop_raster_demo(
            options={
                "raster_path": str(raster_path),
                "output_dir": str(tmp_path / "outputs_tiles"),
                "tile_size": 1,
                "stride": 1,
                "max_tiles": 10,
            }
        )


def test_run_prithvi_crop_raster_demo_rejects_output_dir_outside_allowed_roots(tmp_path: Path):
    from app.services.model_hub_runtime import ModelHubRuntimeError
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

    raster_path = _write_test_geotiff(tmp_path / "crop_18band.tif")
    unsafe_output_dir = repo_root / "ae_backend" / "not_allowed_crop_outputs"

    with pytest.raises(ModelHubRuntimeError, match="output_dir.*allowed"):
        run_prithvi_crop_raster_demo(
            options={"raster_path": str(raster_path), "output_dir": str(unsafe_output_dir)}
        )


def test_run_prithvi_crop_raster_demo_writes_geojson_wgs84_for_projected_raster(tmp_path: Path):
    from app.services.model_hub_crop_raster import run_prithvi_crop_raster_demo

    raster_path = _write_test_geotiff(
        tmp_path / "crop_18band_3857.tif",
        width=6,
        height=6,
        crs="EPSG:3857",
        transform=from_origin(1000000.0, 4000000.0, 30.0, 30.0),
    )

    result = run_prithvi_crop_raster_demo(
        options={
            "raster_path": str(raster_path),
            "output_dir": str(tmp_path / "outputs_3857"),
            "tile_size": 3,
            "stride": 3,
        }
    )

    artifact_by_kind = {artifact["kind"]: Path(artifact["path"]) for artifact in result["artifacts"]}
    geojson = json.loads(artifact_by_kind["geojson"].read_text(encoding="utf-8"))
    manifest = json.loads(artifact_by_kind["manifest"].read_text(encoding="utf-8"))

    assert manifest["geojson_policy"]["crs"] == "EPSG:4326"
    assert manifest["geojson_policy"]["source_crs"] == "EPSG:3857"
    assert manifest["geojson_policy"]["reprojected_to_wgs84"] is True
    first_x, first_y = _first_geojson_coordinate(geojson["features"][0]["geometry"])
    assert -180.0 <= first_x <= 180.0
    assert -90.0 <= first_y <= 90.0
    assert geojson["features"][0]["properties"]["geojson_crs"] == "EPSG:4326"
    assert geojson["features"][0]["properties"]["source_crs"] == "EPSG:3857"
