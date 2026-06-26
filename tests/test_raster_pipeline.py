import sys
from pathlib import Path

import numpy as np


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_make_tile_grid_covers_array_with_overlap():
    from app.services.raster_pipeline import make_tile_grid

    tiles = make_tile_grid(width=300, height=260, tile_size=128, stride=96)

    assert tiles[0] == {"x0": 0, "y0": 0, "x1": 128, "y1": 128}
    assert tiles[-1]["x1"] == 300
    assert tiles[-1]["y1"] == 260
    assert all(tile["x1"] > tile["x0"] and tile["y1"] > tile["y0"] for tile in tiles)


def test_compute_class_area_summary_counts_pixels():
    from app.services.raster_pipeline import compute_class_area_summary

    mask = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)
    summary = compute_class_area_summary(mask, class_names=["background", "built", "water"])

    assert summary["class_pixel_counts"] == {"background": 1, "built": 2, "water": 3}
    assert summary["class_area_fraction"]["water"] == 0.5


def test_stitch_class_tiles_overwrites_expected_window():
    from app.services.raster_pipeline import stitch_class_tiles

    tiles = [
        ({"x0": 0, "y0": 0, "x1": 2, "y1": 2}, np.ones((2, 2), dtype=np.uint8)),
        ({"x0": 1, "y0": 1, "x1": 3, "y1": 3}, np.full((2, 2), 2, dtype=np.uint8)),
    ]
    stitched = stitch_class_tiles(width=3, height=3, tiles=tiles, fill_value=0)

    assert stitched.tolist() == [[1, 1, 0], [1, 2, 2], [0, 2, 2]]
