import sys
from pathlib import Path

import numpy as np


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "scripts"))


def test_ensure_hwc_rgb_converts_chw_patch_layout():
    from run_lulc_api_smoke import ensure_hwc_rgb

    chw = np.zeros((3, 4, 5), dtype=np.uint8)
    chw[0, :, :] = 10
    chw[1, :, :] = 20
    chw[2, :, :] = 30

    hwc = ensure_hwc_rgb(chw)

    assert hwc.shape == (4, 5, 3)
    assert hwc[0, 0].tolist() == [10, 20, 30]


def test_ensure_hwc_rgb_accepts_hwc_patch_layout():
    from run_lulc_api_smoke import ensure_hwc_rgb

    hwc_in = np.zeros((4, 5, 3), dtype=np.uint8)

    hwc = ensure_hwc_rgb(hwc_in)

    assert hwc.shape == (4, 5, 3)


def test_compute_patch_metrics_from_api_prediction():
    from run_lulc_api_smoke import compute_patch_metrics

    prediction = np.array([[2, 2], [3, 0]], dtype=np.int64)
    label = np.array([[2, 2], [3, 3]], dtype=np.int64)

    metrics = compute_patch_metrics(prediction, label, n_classes=6)

    assert metrics["pixel_accuracy"] == 0.75
    assert metrics["per_class_iou"]["class_2"] == 1.0
    assert metrics["per_class_iou"]["class_3"] == 0.5


def test_colorize_uses_linhe_lulc_class_order():
    from run_lulc_api_smoke import PALETTE, colorize

    mask = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)

    rgb = colorize(mask)

    assert rgb.shape == (1, 6, 3)
    assert rgb[0, 0].tolist() == PALETTE[0].tolist()  # background
    assert rgb[0, 1].tolist() == PALETTE[1].tolist()  # built
    assert rgb[0, 4].tolist() == PALETTE[4].tolist()  # water
