from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from affine import Affine
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window

from app.core.config import PROJECT_ROOT
from app.services.model_hub_runtime import ModelHubRuntimeError
from app.services.raster_pipeline import (
    compute_class_area_summary,
    make_tile_grid,
    stitch_class_tiles,
)


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


_ARTIFACT_FILENAMES = {
    "geotiff": "classified_crop.tif",
    "csv": "crop_summary.csv",
    "geojson": "crop_polygons.geojson",
    "manifest": "manifest.json",
    "png": "crop_preview.png",
}
_DEFAULT_MAX_GEOJSON_FEATURES = 5000
_DEFAULT_MAX_PIXELS = 2_000_000
_DEFAULT_MAX_TILES = 4096
_DEFAULT_MAX_PREVIEW_PIXELS = 1_000_000
_MAX_TILE_SIZE = 4096
_MAX_STRIDE = 4096


_CLASS_COLORS = np.array(
    [
        [65, 134, 82],
        [20, 92, 47],
        [239, 197, 74],
        [84, 158, 82],
        [75, 139, 178],
        [172, 151, 124],
        [47, 95, 174],
        [196, 219, 132],
        [96, 184, 126],
        [205, 188, 143],
        [222, 222, 186],
        [190, 117, 74],
        [130, 130, 130],
    ],
    dtype=np.uint8,
)


def _as_jsonable_bounds(bounds: Any) -> list[float]:
    return [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)]


def _default_output_root() -> Path:
    return Path(PROJECT_ROOT) / "results" / "model_hub" / "prithvi_crop_runs"


def _default_output_dir(raster_path: Path) -> Path:
    fingerprint = hashlib.sha256(str(raster_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return _default_output_root() / f"{raster_path.stem}-{fingerprint}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_output_dir(options: dict, raster_path: Path) -> Path:
    output_dir_value = options.get("output_dir")
    if not output_dir_value:
        return _default_output_dir(raster_path)

    output_dir = Path(output_dir_value).expanduser().resolve()
    allowed_roots = [
        _default_output_root().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]
    if not any(output_dir == root or _is_relative_to(output_dir, root) for root in allowed_roots):
        allowed_text = ", ".join(str(root) for root in allowed_roots)
        raise ModelHubRuntimeError(
            f"output_dir must resolve under allowed local demo roots: {allowed_text}"
        )
    return output_dir


def _estimate_tile_count(width: int, height: int, tile_size: int, stride: int) -> int:
    def axis_count(size: int) -> int:
        if size <= tile_size:
            return 1
        last = size - tile_size
        count = (last // stride) + 1
        if last % stride != 0:
            count += 1
        return count

    return axis_count(width) * axis_count(height)


def _class_id(class_name: str) -> int:
    return CROP_RASTER_CLASSES.index(class_name)


def _parse_positive_int_option(options: dict, name: str, default: int) -> int:
    raw_value = options.get(name, default)
    if isinstance(raw_value, bool):
        raise ModelHubRuntimeError(f"{name} must be a positive integer")
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ModelHubRuntimeError(f"{name} must be a positive integer") from exc
    if isinstance(raw_value, float) and not raw_value.is_integer():
        raise ModelHubRuntimeError(f"{name} must be a positive integer")
    if parsed < 1:
        raise ModelHubRuntimeError(f"{name} must be at least 1")
    return parsed


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)


def _classify_tile(tile: np.ndarray, *, tile_index: int) -> np.ndarray:
    tile_data = np.asarray(tile, dtype=np.float32)
    if tile_data.ndim != 3 or tile_data.shape[0] < 18:
        raise ModelHubRuntimeError("Prithvi crop tile classification requires 18 bands")

    blue = tile_data[[0, 6, 12]].mean(axis=0)
    green = tile_data[[1, 7, 13]].mean(axis=0)
    red = tile_data[[2, 8, 14]].mean(axis=0)
    nir = tile_data[[3, 9, 15]].mean(axis=0)
    swir = tile_data[[4, 5, 10, 11, 16, 17]].mean(axis=0)
    ndvi = _safe_ratio(nir - red, nir + red)
    ndwi = _safe_ratio(green - nir, green + nir)
    brightness = (blue + green + red + nir + swir) / 5.0

    rows, cols = np.indices(red.shape)
    spatial_score = rows * 3 + cols * 5 + tile_index * 7
    spectral_score = np.floor((brightness + ndvi + 1.0) * 100.0).astype(np.int32)
    class_mask = ((spectral_score + spatial_score) % len(CROP_RASTER_CLASSES)).astype(np.uint8)

    class_mask = np.where(ndwi > 0.2, _class_id("open_water"), class_mask)
    class_mask = np.where((ndvi > 0.35) & (swir < nir), _class_id("forest"), class_mask)
    class_mask = np.where((ndvi > 0.2) & (brightness > 0.35), _class_id("corn"), class_mask)
    class_mask = np.where((brightness > 0.7) & (ndvi < 0.12), _class_id("developed_barren"), class_mask)
    return class_mask.astype(np.uint8)


def _write_classified_geotiff(
    output_path: Path,
    *,
    mask: np.ndarray,
    source_profile: dict,
) -> None:
    profile = dict(source_profile)
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
    )
    if not profile.get("tiled"):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def _write_summary_csv(output_path: Path, summary: dict) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=["class", "pixels", "fraction"])
        writer.writeheader()
        for class_name in CROP_RASTER_CLASSES:
            writer.writerow(
                {
                    "class": class_name,
                    "pixels": summary["class_pixel_counts"][class_name],
                    "fraction": f'{summary["class_area_fraction"][class_name]:.6f}',
                }
            )


def _write_geojson(
    output_path: Path,
    *,
    mask: np.ndarray,
    transform: Affine,
    source_crs: Any,
    max_features: int,
) -> dict:
    source_crs_text = source_crs.to_string() if hasattr(source_crs, "to_string") else str(source_crs)
    geojson_crs = "EPSG:4326"
    reproject_to_wgs84 = source_crs_text != geojson_crs
    features_written = 0
    features_truncated = False
    with output_path.open("w", encoding="utf-8") as geojson_file:
        geojson_file.write('{"type":"FeatureCollection","features":[')
        for feature_index, (geometry, value) in enumerate(
            shapes(mask.astype(np.uint8), transform=transform)
        ):
            if feature_index >= max_features:
                features_truncated = True
                break
            if reproject_to_wgs84:
                geometry = transform_geom(source_crs_text, geojson_crs, geometry, precision=6)
            class_id = int(value)
            feature = {
                "type": "Feature",
                "properties": {
                    "class_id": class_id,
                    "class_name": CROP_RASTER_CLASSES[class_id],
                    "source_crs": source_crs_text,
                    "geojson_crs": geojson_crs,
                },
                "geometry": geometry,
            }
            if features_written:
                geojson_file.write(",")
            json.dump(feature, geojson_file, separators=(",", ":"))
            features_written += 1
        geojson_file.write("]}")
    return {
        "strategy": "streaming_feature_limit",
        "crs": geojson_crs,
        "source_crs": source_crs_text,
        "reprojected_to_wgs84": reproject_to_wgs84,
        "max_features": max_features,
        "features_written": features_written,
        "features_truncated": features_truncated,
    }


def _write_preview_png(output_path: Path, *, mask: np.ndarray) -> None:
    from PIL import Image

    preview = _CLASS_COLORS[mask.astype(np.uint8)]
    Image.fromarray(preview, mode="RGB").save(output_path)


def _write_manifest(output_path: Path, *, manifest: dict) -> None:
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
            if src.transform is None or src.transform == Affine.identity():
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


def run_prithvi_crop_raster_demo(*, options: dict) -> dict:
    raster_path_value = options.get("raster_path")
    if not raster_path_value:
        raise ModelHubRuntimeError("raster_path is required for upload_raster_demo")

    raster_path = Path(raster_path_value)
    validation = validate_prithvi_crop_raster(raster_path)
    pixel_count = int(validation["width"] * validation["height"])
    max_pixels = _parse_positive_int_option(options, "max_pixels", _DEFAULT_MAX_PIXELS)
    if pixel_count > max_pixels:
        raise ModelHubRuntimeError(
            f"Prithvi crop raster has {pixel_count} pixels, exceeds max_pixels={max_pixels}"
        )

    output_dir = _resolve_output_dir(options, raster_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = _parse_positive_int_option(options, "tile_size", 224)
    stride = _parse_positive_int_option(options, "stride", tile_size)
    if tile_size > _MAX_TILE_SIZE:
        raise ModelHubRuntimeError(f"tile_size must be at most {_MAX_TILE_SIZE}")
    if stride > _MAX_STRIDE:
        raise ModelHubRuntimeError(f"stride must be at most {_MAX_STRIDE}")
    max_tiles = _parse_positive_int_option(options, "max_tiles", _DEFAULT_MAX_TILES)
    max_preview_pixels = _parse_positive_int_option(
        options,
        "max_preview_pixels",
        _DEFAULT_MAX_PREVIEW_PIXELS,
    )
    max_geojson_features = _parse_positive_int_option(
        options,
        "max_geojson_features",
        _DEFAULT_MAX_GEOJSON_FEATURES,
    )
    logs = [
        f"validated 18-band Prithvi crop raster from {raster_path}",
        "using deterministic tiled crop classification; no real Prithvi checkpoint was loaded",
    ]

    estimated_tile_count = _estimate_tile_count(
        validation["width"],
        validation["height"],
        tile_size,
        stride,
    )
    if estimated_tile_count > max_tiles:
        raise ModelHubRuntimeError(
            f"Prithvi crop raster requires {estimated_tile_count} tiles, exceeds max_tiles={max_tiles}"
        )

    with rasterio.open(raster_path) as src:
        tiles = make_tile_grid(src.width, src.height, tile_size, stride)
        tile_masks = []
        for tile_index, window in enumerate(tiles):
            raster_window = Window(
                col_off=window["x0"],
                row_off=window["y0"],
                width=window["x1"] - window["x0"],
                height=window["y1"] - window["y0"],
            )
            tile = src.read(window=raster_window, boundless=False)
            tile_masks.append((window, _classify_tile(tile, tile_index=tile_index)))
        mask = stitch_class_tiles(
            width=src.width,
            height=src.height,
            tiles=tile_masks,
            fill_value=_class_id("other"),
        )
        source_profile = dict(src.profile)
        source_transform = src.transform
        source_crs = src.crs

    summary = compute_class_area_summary(mask, CROP_RASTER_CLASSES)
    dominant_class = max(summary["class_pixel_counts"], key=summary["class_pixel_counts"].get)
    summary["dominant_class"] = dominant_class
    summary["method"] = "deterministic tiled crop classification demo"
    summary["tile_count"] = len(tiles)

    classified_tif = output_dir / _ARTIFACT_FILENAMES["geotiff"]
    summary_csv = output_dir / _ARTIFACT_FILENAMES["csv"]
    polygons_geojson = output_dir / _ARTIFACT_FILENAMES["geojson"]
    manifest_json = output_dir / _ARTIFACT_FILENAMES["manifest"]
    preview_png = output_dir / _ARTIFACT_FILENAMES["png"]

    artifacts = [
        {"kind": "geotiff", "path": str(classified_tif)},
        {"kind": "csv", "path": str(summary_csv)},
        {"kind": "geojson", "path": str(polygons_geojson)},
        {"kind": "manifest", "path": str(manifest_json)},
    ]

    _write_classified_geotiff(classified_tif, mask=mask, source_profile=source_profile)
    _write_summary_csv(summary_csv, summary)
    geojson_policy = _write_geojson(
        polygons_geojson,
        mask=mask,
        transform=source_transform,
        source_crs=source_crs,
        max_features=max_geojson_features,
    )
    logs.append(
        "GeoJSON feature limit "
        f"{geojson_policy['max_features']} wrote {geojson_policy['features_written']} "
        f"features; truncated={geojson_policy['features_truncated']}"
    )
    if mask.size > max_preview_pixels:
        logs.append(
            f"skipped PNG preview at {preview_png}: {mask.size} pixels exceeds "
            f"max_preview_pixels={max_preview_pixels}"
        )
    else:
        try:
            _write_preview_png(preview_png, mask=mask)
        except (ImportError, OSError, ValueError) as exc:
            logs.append(f"skipped PNG preview at {preview_png}: {exc}")
        else:
            artifacts.append({"kind": "png", "path": str(preview_png)})

    model_package = {
        "package_type": "arcgis_style_pretrained_imagery_model",
        "family": "prithvi_crop_classification",
        "runtime_mode": "upload_raster_demo",
        "class_schema": list(CROP_RASTER_CLASSES),
    }
    result = {
        "task": "crop_classification",
        "model_id": CROP_RASTER_MODEL_ID,
        "input_mode": "upload_raster_demo",
        "validation": validation,
        "summary": summary,
        "model_package": model_package,
    }
    manifest = {
        "model_id": CROP_RASTER_MODEL_ID,
        "input_mode": "upload_raster_demo",
        "source_raster": str(raster_path),
        "validation": validation,
        "tile_grid": {
            "tile_size": tile_size,
            "stride": stride,
            "tile_count": len(tiles),
            "overlap_policy": "last_tile_wins" if stride < tile_size else "none",
        },
        "resource_policy": {
            "pixel_count": pixel_count,
            "max_pixels": max_pixels,
            "tile_count": len(tiles),
            "max_tiles": max_tiles,
            "max_preview_pixels": max_preview_pixels,
        },
        "geojson_policy": geojson_policy,
        "artifacts": artifacts,
        "limitations": [
            "Deterministic demo classification only.",
            "No real Prithvi checkpoint was loaded.",
            "Class IDs are generated from lightweight spectral and spatial rules.",
            "GeoJSON polygons are streamed, capped by max_geojson_features, and written in EPSG:4326.",
        ],
    }
    _write_manifest(manifest_json, manifest=manifest)

    return {"result": result, "artifacts": artifacts, "logs": logs}
