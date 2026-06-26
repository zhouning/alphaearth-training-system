from __future__ import annotations

import numpy as np


def _starts(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]
    starts = list(range(0, max(size - tile_size, 0) + 1, stride))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tile_grid(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
) -> list[dict[str, int]]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    tiles: list[dict[str, int]] = []
    for y0 in _starts(height, tile_size, stride):
        for x0 in _starts(width, tile_size, stride):
            tiles.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(min(x0 + tile_size, width)),
                    "y1": int(min(y0 + tile_size, height)),
                }
            )
    return tiles


def compute_class_area_summary(mask: np.ndarray, class_names: list[str]) -> dict:
    mask_array = np.asarray(mask)
    total = int(mask_array.size)
    denominator = max(total, 1)
    counts: dict[str, int] = {}
    fractions: dict[str, float] = {}
    for class_id, class_name in enumerate(class_names):
        count = int(np.count_nonzero(mask_array == class_id))
        counts[class_name] = count
        fractions[class_name] = count / denominator
    return {"class_pixel_counts": counts, "class_area_fraction": fractions}


def stitch_class_tiles(
    *,
    width: int,
    height: int,
    tiles: list[tuple[dict[str, int], np.ndarray]],
    fill_value: int = 0,
) -> np.ndarray:
    stitched = np.full((height, width), fill_value, dtype=np.uint8)
    for window, tile_mask in tiles:
        y0, y1 = int(window["y0"]), int(window["y1"])
        x0, x1 = int(window["x0"]), int(window["x1"])
        stitched[y0:y1, x0:x1] = np.asarray(
            tile_mask,
            dtype=np.uint8,
        )[: y1 - y0, : x1 - x0]
    return stitched
