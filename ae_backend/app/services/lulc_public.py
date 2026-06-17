from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


PUBLIC_LULC_CLASS_NAMES = [
    "background",
    "built",
    "crops",
    "trees",
    "water",
    "rangeland_bare",
]

ESRI_9_TO_LINHE_6 = {
    0: 0,
    1: 4,
    2: 3,
    4: 4,
    5: 2,
    7: 1,
    8: 5,
    9: 0,
    10: 0,
    11: 5,
}


class PublicLULCNotAvailableError(ValueError):
    """Raised when a requested public LULC product is unavailable locally."""


def summarize_public_mask(mask: np.ndarray) -> dict[str, dict[str, int | float]]:
    mask_array = np.asarray(mask)
    total_pixels = int(mask_array.size)
    denominator = max(total_pixels, 1)

    counts: dict[str, int] = {}
    fractions: dict[str, float] = {}
    for class_id, class_name in enumerate(PUBLIC_LULC_CLASS_NAMES):
        count = int(np.count_nonzero(mask_array == class_id))
        counts[class_name] = count
        fractions[class_name] = count / denominator
    return {
        "class_pixel_counts": counts,
        "class_area_fraction": fractions,
    }


def remap_esri_9_to_linhe_6(mask: np.ndarray) -> np.ndarray:
    """Map Esri native 9-class codes to the Linhe 6-class schema."""
    raw = np.asarray(mask)
    out = np.zeros(raw.shape, dtype=np.int64)
    for source_id, target_id in ESRI_9_TO_LINHE_6.items():
        out[raw == source_id] = target_id
    return out


class CachedRasterLULCProvider:
    """Serve public LULC product rasters already cached as local GeoTIFFs."""

    provider_id = "esri_lulc_cache"
    product_id = "esri_impact_observatory_10m_lulc"

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def available_years(self) -> list[int]:
        if not self.cache_dir.exists():
            return []
        years: list[int] = []
        for path in self.cache_dir.glob("*.tif"):
            try:
                years.append(int(path.stem))
            except ValueError:
                continue
        return sorted(set(years))

    def _raster_for_year(self, year: int) -> Path:
        raster_path = self.cache_dir / f"{year}.tif"
        if not raster_path.exists():
            available = self.available_years()
            suffix = f" Available years: {available}." if available else ""
            raise PublicLULCNotAvailableError(
                f"Public LULC cache is missing year {year}: {raster_path}.{suffix}"
            )
        return raster_path

    def query(
        self,
        *,
        year: int,
        bbox: tuple[float, float, float, float],
        bbox_crs: str = "EPSG:4326",
    ) -> dict:
        raster_path = self._raster_for_year(year)
        with rasterio.open(raster_path) as src:
            if bbox_crs != src.crs.to_string():
                read_bounds = transform_bounds(bbox_crs, src.crs, *bbox, densify_pts=21)
            else:
                read_bounds = bbox
            dataset_bounds = src.bounds
            minx = max(read_bounds[0], dataset_bounds.left)
            miny = max(read_bounds[1], dataset_bounds.bottom)
            maxx = min(read_bounds[2], dataset_bounds.right)
            maxy = min(read_bounds[3], dataset_bounds.top)
            if minx >= maxx or miny >= maxy:
                raise PublicLULCNotAvailableError(
                    "Requested bbox does not overlap the cached public LULC raster."
                )

            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            raw_mask = src.read(1, window=window, boundless=False).astype(np.int64)
            mask = remap_esri_9_to_linhe_6(raw_mask)
            resolution = abs(float(src.res[0])) if src.res else None
            out_bounds = [float(minx), float(miny), float(maxx), float(maxy)]
            raster_crs = src.crs.to_string()

        summary = summarize_public_mask(mask)
        return {
            "task": "lulc_segmentation",
            "mode": "public_product",
            "provider_id": self.provider_id,
            "product_id": self.product_id,
            "year": year,
            "classes": PUBLIC_LULC_CLASS_NAMES,
            "class_schema": "linhe_esri_6class_public_cache",
            "source_class_schema": "esri_native_9class",
            "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "mask": mask.tolist(),
            "native_resolution_m": resolution,
            "bbox": out_bounds,
            "bbox_crs": raster_crs,
            "source": {
                "type": "local_geotiff_cache",
                "path": str(raster_path),
                "upstream": "Impact Observatory / Esri / Microsoft 10m annual LULC",
            },
            "limitations": [
                "This is a public 10 m land-cover product, not an AlphaEarth model prediction.",
                "Class schema differs from the local RGB checkpoint schema.",
            ],
            **summary,
        }


def query_public_lulc(
    *,
    provider_id: str,
    year: int,
    bbox: tuple[float, float, float, float],
    bbox_crs: str,
    cache_dir: str | Path,
) -> dict:
    if provider_id != CachedRasterLULCProvider.provider_id:
        raise PublicLULCNotAvailableError(
            f"Provider {provider_id!r} is not queryable in this local API. "
            "Use 'esri_lulc_cache' for cached public rasters."
        )
    return CachedRasterLULCProvider(cache_dir=cache_dir).query(
        year=year,
        bbox=bbox,
        bbox_crs=bbox_crs,
    )
