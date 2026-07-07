from __future__ import annotations

from pathlib import Path
from typing import Any

from affine import Affine
import rasterio
from rasterio.errors import RasterioIOError

from app.core.config import PROJECT_ROOT
from app.services.model_hub_runtime import ModelHubRuntimeError


FLOOD_MODEL_ID = "water_flood_prithvi"
FLOOD_BAND_ORDER = ["blue", "green", "red", "narrow_nir", "swir1", "swir2"]


def _default_weights_dir() -> Path:
    return Path(PROJECT_ROOT) / "data" / "weights" / "prithvi_flood"


def _as_jsonable_bounds(bounds: Any) -> list[float]:
    return [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)]


def _has_weight_files(weights_dir: Path) -> bool:
    if not weights_dir.exists():
        return False
    if weights_dir.is_file():
        return weights_dir.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    return any(
        child.is_file() and child.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
        for child in weights_dir.rglob("*")
    )


def validate_flood_raster(raster_path: str | Path) -> dict:
    path = Path(raster_path)
    if not path.exists():
        raise ModelHubRuntimeError(f"Flood raster does not exist: {path}")
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ModelHubRuntimeError("Flood raster must be a GeoTIFF .tif or .tiff file")

    try:
        with rasterio.open(path) as src:
            if src.count != 6:
                raise ModelHubRuntimeError(
                    f"Flood raster requires 6 bands, got {src.count}"
                )
            if src.width <= 0 or src.height <= 0:
                raise ModelHubRuntimeError("Flood raster width and height must be positive")
            if src.crs is None:
                raise ModelHubRuntimeError("Flood raster requires georeferencing CRS")
            if src.transform is None or src.transform == Affine.identity():
                raise ModelHubRuntimeError("Flood raster requires georeferencing transform")
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
                "band_order": list(FLOOD_BAND_ORDER),
            }
    except RasterioIOError as exc:
        raise ModelHubRuntimeError(f"Could not open flood raster: {path}") from exc


def _require_flood_runtime(weights_dir: Path) -> None:
    if not _has_weight_files(weights_dir):
        raise ModelHubRuntimeError(
            f"Prithvi flood weights are missing at {weights_dir}. "
            "Use scripts/model_hub/fetch_public_sample.py --asset prithvi_flood --dry-run "
            "to inspect the approved public source before downloading weights."
        )
    try:
        import terratorch  # noqa: F401
    except ImportError as exc:
        raise ModelHubRuntimeError(
            "Prithvi flood real inference requires TerraTorch-compatible runtime dependencies."
        ) from exc


def run_real_flood_inference(*, options: dict) -> dict:
    raster_path = options.get("raster_path")
    if not raster_path:
        raise ModelHubRuntimeError("raster_path is required for real flood inference")

    validation = validate_flood_raster(raster_path)
    weights_dir = Path(options.get("weights_dir") or _default_weights_dir())
    _require_flood_runtime(weights_dir)
    raise ModelHubRuntimeError(
        "Prithvi flood neural runtime passed input and asset guards, but the "
        "TerraTorch inference adapter has not completed local verification. "
        f"validation={validation['band_count']} bands, model_id={FLOOD_MODEL_ID}."
    )
