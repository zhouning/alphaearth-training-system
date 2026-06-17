import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_cached_raster_lulc_provider_reads_bbox_and_summarizes_classes(tmp_path: Path):
    from app.services.lulc_public import CachedRasterLULCProvider

    raster_path = tmp_path / "2022.tif"
    values = np.array(
        [
            [5, 5, 7, 7],
            [5, 2, 7, 1],
            [11, 11, 1, 1],
            [8, 0, 0, 4],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 40, 10, 10),
    ) as dst:
        dst.write(values, 1)

    provider = CachedRasterLULCProvider(cache_dir=tmp_path)
    result = provider.query(
        year=2022,
        bbox=(0, 0, 40, 40),
        bbox_crs="EPSG:3857",
    )

    assert result["mode"] == "public_product"
    assert result["provider_id"] == "esri_lulc_cache"
    assert result["product_id"] == "esri_impact_observatory_10m_lulc"
    assert result["year"] == 2022
    assert result["mask_shape"] == [4, 4]
    assert result["class_pixel_counts"]["crops"] == 3
    assert result["class_pixel_counts"]["built"] == 3
    assert result["class_pixel_counts"]["water"] == 4
    assert result["class_pixel_counts"]["rangeland_bare"] == 3
    assert result["class_pixel_counts"]["background"] == 2
    assert result["source_class_schema"] == "esri_native_9class"
    assert result["native_resolution_m"] == 10
    assert result["source"]["type"] == "local_geotiff_cache"


def test_cached_raster_lulc_provider_rejects_missing_year(tmp_path: Path):
    from app.services.lulc_public import CachedRasterLULCProvider, PublicLULCNotAvailableError

    provider = CachedRasterLULCProvider(cache_dir=tmp_path)

    with pytest.raises(PublicLULCNotAvailableError, match="2022"):
        provider.query(year=2022, bbox=(0, 0, 1, 1), bbox_crs="EPSG:3857")
