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
