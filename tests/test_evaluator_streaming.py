import numpy as np
import pytest

from geoadapter.engine.evaluator import (
    SegmentationConfusionMatrix,
    compute_segmentation_metrics,
)


class TestSegmentationConfusionMatrix:
    def test_streaming_equals_batch(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 7, size=(8, 64, 64))
        y_pred = rng.randint(0, 7, size=(8, 64, 64))
        batch = compute_segmentation_metrics(y_true, y_pred)
        cm = SegmentationConfusionMatrix(ignore_index=255)
        for i in range(y_true.shape[0]):
            cm.update(y_true[i], y_pred[i])
        streaming = cm.compute()
        assert streaming["mIoU"] == pytest.approx(batch["mIoU"], abs=1e-9)

    def test_ignore_index_is_excluded(self):
        y_true = np.array([[0, 1, 255, 255], [255, 255, 255, 255]])
        y_pred = np.array([[0, 1, 0, 0], [3, 3, 3, 3]])
        cm = SegmentationConfusionMatrix(ignore_index=255)
        cm.update(y_true, y_pred)
        m = cm.compute()
        assert m["mIoU"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_prediction(self):
        y = np.array([[0, 1, 2], [3, 4, 5]])
        cm = SegmentationConfusionMatrix(ignore_index=255)
        cm.update(y, y)
        assert cm.compute()["mIoU"] == pytest.approx(1.0, abs=1e-9)

    def test_completely_wrong(self):
        y_true = np.zeros((4, 4), dtype=np.int64)
        y_pred = np.ones((4, 4), dtype=np.int64)
        cm = SegmentationConfusionMatrix(ignore_index=255)
        cm.update(y_true, y_pred)
        assert cm.compute()["mIoU"] == pytest.approx(0.0, abs=1e-9)

    def test_handles_class_outside_inferred_range(self):
        cm = SegmentationConfusionMatrix(num_classes=7, ignore_index=255)
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 99])
        cm.update(y_true, y_pred)
        m = cm.compute()
        # 99 is outside [0, num_classes); the (true=2, pred=99) pixel is
        # silently dropped, leaving (0,0) and (1,1) — both correct → mIoU=1.0.
        assert m["mIoU"] == pytest.approx(1.0, abs=1e-9)
        # The dropped pixel must not poison the confusion matrix indices.
        assert cm.cm.shape == (7, 7)
        assert cm.cm[2, :].sum() == 0  # class 2 had only the dropped pixel

    def test_empty_after_ignore_does_not_crash(self):
        cm = SegmentationConfusionMatrix(ignore_index=255)
        y_true = np.full((4, 4), 255)
        y_pred = np.zeros((4, 4))
        cm.update(y_true, y_pred)
        assert cm.compute()["mIoU"] == pytest.approx(0.0, abs=1e-9)

    def test_per_class_iou_and_pred_histogram_for_collapse_diagnosis(self):
        # Collapse case: model predicts class 0 everywhere; class 0 is 60% of GT
        # (mirrors LoveDA bg distribution). Per-class IoU and pred_pixel_count
        # are what Step 8.5 of the LoveDA notebook uses to detect majority-class
        # collapse separately from raw mIoU.
        cm = SegmentationConfusionMatrix(num_classes=3, ignore_index=255)
        y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
        y_pred = np.zeros_like(y_true)
        cm.update(y_true, y_pred)
        m = cm.compute()
        assert m["per_class_iou"][0] == pytest.approx(0.6, abs=1e-9)
        assert m["per_class_iou"][1] == pytest.approx(0.0, abs=1e-9)
        assert m["per_class_iou"][2] == pytest.approx(0.0, abs=1e-9)
        assert m["gt_pixel_count"] == [6, 2, 2]
        assert m["pred_pixel_count"] == [10, 0, 0]
