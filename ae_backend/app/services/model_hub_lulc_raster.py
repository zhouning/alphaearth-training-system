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


LULC_RASTER_MODEL_ID = "lulc_6class_prithvi_houlsby"
LULC_CLASSES = ["background", "built", "crops", "trees", "water", "rangeland_bare"]
LULC_BAND_ORDER = ["red", "green", "blue"]

_ARTIFACT_FILENAMES = {
    "geotiff": "classified_lulc.tif",
    "csv": "lulc_summary.csv",
    "geojson": "lulc_polygons.geojson",
    "manifest": "manifest.json",
    "png": "lulc_preview.png",
}
_DEFAULT_MAX_GEOJSON_FEATURES = 5000
_DEFAULT_MAX_PIXELS = 2_000_000
_DEFAULT_MAX_TILES = 4096
_DEFAULT_MAX_PREVIEW_PIXELS = 1_000_000
_MAX_TILE_SIZE = 4096
_MAX_STRIDE = 4096

_CLASS_COLORS = np.array(
    [
        [34, 34, 34],
        [203, 85, 58],
        [238, 201, 74],
        [72, 156, 83],
        [64, 121, 197],
        [154, 139, 96],
    ],
    dtype=np.uint8,
)


def _default_input_root() -> Path:
    return Path(PROJECT_ROOT) / "results" / "model_hub" / "lulc_inputs"


def _default_output_root() -> Path:
    return Path(PROJECT_ROOT) / "results" / "model_hub" / "lulc_runs"


def _default_output_dir(raster_path: Path) -> Path:
    fingerprint = hashlib.sha256(str(raster_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return _default_output_root() / f"{raster_path.stem}-{fingerprint}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _as_jsonable_bounds(bounds: Any) -> list[float]:
    return [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)]


def _resolve_input_raster_path(raster_path_value: str | Path) -> Path:
    raw_path = Path(raster_path_value).expanduser()
    if not raw_path.is_absolute():
        raw_path = Path(PROJECT_ROOT) / raw_path
    raster_path = raw_path.resolve()
    allowed_roots = [
        Path(PROJECT_ROOT).resolve(),
        _default_input_root().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]
    if not any(raster_path == root or _is_relative_to(raster_path, root) for root in allowed_roots):
        allowed_text = ", ".join(str(root) for root in allowed_roots)
        raise ModelHubRuntimeError(
            f"raster_path must resolve under allowed local roots: {allowed_text}"
        )
    return raster_path


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
            f"output_dir must resolve under allowed local roots: {allowed_text}"
        )
    return output_dir


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


def _parse_capped_positive_int_option(
    options: dict,
    name: str,
    default: int,
    maximum: int,
) -> int:
    parsed = _parse_positive_int_option(options, name, default)
    if parsed > maximum:
        raise ModelHubRuntimeError(f"{name} must be at most {maximum}")
    return parsed


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


def _write_classified_geotiff(
    output_path: Path,
    *,
    mask: np.ndarray,
    source_profile: dict,
) -> None:
    profile = dict(source_profile)
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate")
    if not profile.get("tiled"):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def _write_summary_csv(output_path: Path, summary: dict, class_names: list[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=["class", "pixels", "fraction"])
        writer.writeheader()
        for class_name in class_names:
            writer.writerow(
                {
                    "class": class_name,
                    "pixels": summary["class_pixel_counts"].get(class_name, 0),
                    "fraction": f'{summary["class_area_fraction"].get(class_name, 0.0):.6f}',
                }
            )


def _write_geojson(
    output_path: Path,
    *,
    mask: np.ndarray,
    transform: Affine,
    source_crs: Any,
    class_names: list[str],
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
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "unknown"
            feature = {
                "type": "Feature",
                "properties": {
                    "class_id": class_id,
                    "class_name": class_name,
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

    safe_mask = np.clip(mask.astype(np.int64), 0, len(_CLASS_COLORS) - 1)
    preview = _CLASS_COLORS[safe_mask]
    Image.fromarray(preview, mode="RGB").save(output_path)


def _write_manifest(output_path: Path, *, manifest: dict) -> None:
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _predict_rgb_tile(
    image: np.ndarray,
    model_id: str | None,
    checkpoint_path: str | None,
) -> dict:
    from app.api.inference import get_lulc_service

    service = get_lulc_service(checkpoint_path=checkpoint_path, model_id=model_id)
    return service.predict_image(image)


def validate_lulc_raster(raster_path: str | Path) -> dict:
    path = Path(raster_path)
    if not path.exists():
        raise ModelHubRuntimeError(f"LULC raster does not exist: {path}")
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ModelHubRuntimeError("LULC raster must be a GeoTIFF .tif or .tiff file")

    try:
        with rasterio.open(path) as src:
            if src.count != 3:
                raise ModelHubRuntimeError(f"LULC raster requires 3 bands, got {src.count}")
            if src.width <= 0 or src.height <= 0:
                raise ModelHubRuntimeError("LULC raster width and height must be positive")
            if src.crs is None:
                raise ModelHubRuntimeError("LULC raster requires georeferencing CRS")
            if src.transform is None or src.transform == Affine.identity():
                raise ModelHubRuntimeError("LULC raster requires georeferencing transform")
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
                "band_order": list(LULC_BAND_ORDER),
            }
    except RasterioIOError as exc:
        raise ModelHubRuntimeError(f"Could not open LULC raster: {path}") from exc


def run_lulc_raster_inference(*, options: dict) -> dict:
    raster_path_value = options.get("raster_path")
    if not raster_path_value:
        raise ModelHubRuntimeError("raster_path is required for LULC raster inference")

    raster_path = _resolve_input_raster_path(raster_path_value)
    validation = validate_lulc_raster(raster_path)
    pixel_count = int(validation["width"] * validation["height"])
    max_pixels = _parse_capped_positive_int_option(
        options, "max_pixels", _DEFAULT_MAX_PIXELS, _DEFAULT_MAX_PIXELS
    )
    if pixel_count > max_pixels:
        raise ModelHubRuntimeError(
            f"LULC raster has {pixel_count} pixels, exceeds max_pixels={max_pixels}"
        )

    output_dir = _resolve_output_dir(options, raster_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = _parse_capped_positive_int_option(options, "tile_size", 128, _MAX_TILE_SIZE)
    stride = _parse_capped_positive_int_option(options, "stride", tile_size, _MAX_STRIDE)
    if stride > tile_size:
        raise ModelHubRuntimeError("stride must be less than or equal to tile_size")
    max_tiles = _parse_capped_positive_int_option(
        options, "max_tiles", _DEFAULT_MAX_TILES, _DEFAULT_MAX_TILES
    )
    max_preview_pixels = _parse_capped_positive_int_option(
        options,
        "max_preview_pixels",
        _DEFAULT_MAX_PREVIEW_PIXELS,
        _DEFAULT_MAX_PREVIEW_PIXELS,
    )
    max_geojson_features = _parse_capped_positive_int_option(
        options,
        "max_geojson_features",
        _DEFAULT_MAX_GEOJSON_FEATURES,
        _DEFAULT_MAX_GEOJSON_FEATURES,
    )
    estimated_tile_count = _estimate_tile_count(
        validation["width"],
        validation["height"],
        tile_size,
        stride,
    )
    if estimated_tile_count > max_tiles:
        raise ModelHubRuntimeError(
            f"LULC raster requires {estimated_tile_count} tiles, exceeds max_tiles={max_tiles}"
        )

    model_id = options.get("model_id") or LULC_RASTER_MODEL_ID
    checkpoint_path = options.get("checkpoint_path")
    logs = [
        f"validated 3-band LULC raster from {raster_path}",
        "using local LULC checkpoint tiled inference",
    ]

    with rasterio.open(raster_path) as src:
        tiles = make_tile_grid(src.width, src.height, tile_size, stride)
        tile_masks = []
        classes = list(LULC_CLASSES)
        for window in tiles:
            raster_window = Window(
                col_off=window["x0"],
                row_off=window["y0"],
                width=window["x1"] - window["x0"],
                height=window["y1"] - window["y0"],
            )
            tile = src.read(window=raster_window, boundless=False)
            image = np.moveaxis(tile, 0, -1)
            prediction = _predict_rgb_tile(image, model_id, checkpoint_path)
            classes = list(prediction.get("classes") or classes)
            tile_mask = np.asarray(prediction["mask"], dtype=np.uint8)
            tile_masks.append((window, tile_mask))
        mask = stitch_class_tiles(width=src.width, height=src.height, tiles=tile_masks)
        source_profile = dict(src.profile)
        source_transform = src.transform
        source_crs = src.crs

    summary = compute_class_area_summary(mask, classes)
    dominant_class = max(summary["class_pixel_counts"], key=summary["class_pixel_counts"].get)
    summary["dominant_class"] = dominant_class
    summary["method"] = "local LULC checkpoint tiled inference"
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
    _write_summary_csv(summary_csv, summary, classes)
    geojson_policy = _write_geojson(
        polygons_geojson,
        mask=mask,
        transform=source_transform,
        source_crs=source_crs,
        class_names=classes,
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

    result = {
        "task": "lulc_segmentation",
        "model_id": model_id,
        "input_mode": "raster_inference",
        "validation": validation,
        "summary": summary,
        "class_schema": classes,
        "runtime_kind": "neural_checkpoint",
    }
    manifest = {
        "model_id": model_id,
        "input_mode": "raster_inference",
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
            "Smoke-test production wrapper; accuracy depends on the selected local checkpoint.",
            "GeoJSON polygons are streamed, capped by max_geojson_features, and written in EPSG:4326.",
        ],
    }
    _write_manifest(manifest_json, manifest=manifest)

    return {"result": result, "artifacts": artifacts, "logs": logs}
