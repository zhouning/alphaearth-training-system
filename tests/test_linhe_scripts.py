"""Unit tests for Linhe script utilities."""

import numpy as np

from scripts.linhe_build_patches import tile_array
from scripts.linhe_finetune import build_pseudo_labels


def test_tile_array_skips_mostly_zero_tiles():
    arr = np.zeros((3, 256, 256), dtype=np.uint8)
    arr[:, 0:128, 0:128] = 255

    result = tile_array(arr, patch=128, stride=128)

    assert len(result) == 1
    row, col, _ = result[0]
    assert row == 0
    assert col == 0


def test_build_pseudo_labels_returns_one_label_per_patch(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"patch_{i}.npz"
        rgb = np.full((3, 128, 128), i * 40, dtype=np.uint8)
        np.savez(p, rgb=rgb)
        paths.append(p)

    labels = build_pseudo_labels(paths, n_clusters=2)

    assert len(labels) == 4
    assert set(labels.tolist()) <= {0, 1}
