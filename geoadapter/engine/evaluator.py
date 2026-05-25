import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score


def compute_classification_metrics(y_true, y_pred):
    """Compute OA and Macro F1 for single-label classification."""
    return {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def compute_multilabel_metrics(y_true, y_scores):
    """Compute mAP for multi-label classification."""
    return {
        "mAP": average_precision_score(y_true, y_scores, average="macro"),
    }


def compute_segmentation_metrics(y_true, y_pred, ignore_index=255):
    """Compute mean IoU for semantic segmentation (batch-mode, kept for compatibility)."""
    acc = SegmentationConfusionMatrix(ignore_index=ignore_index)
    acc.update(y_true, y_pred)
    return acc.compute()


class SegmentationConfusionMatrix:
    """Streaming confusion matrix for semantic segmentation.

    Lets us compute mIoU over arbitrarily large val sets (LoveDA: 992 images at
    1024x1024 = ~16 GB worth of int64 if stockpiled in RAM) by accumulating an
    N_classes x N_classes int64 matrix per batch instead. N_classes is inferred
    from data on the first update.
    """

    def __init__(self, num_classes=None, ignore_index=255):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.cm = None

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        mask = y_true != self.ignore_index
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_true.size == 0:
            return
        if self.num_classes is None:
            self.num_classes = int(max(y_true.max(), y_pred.max())) + 1
            self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        elif self.cm is None:
            self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        nc = self.num_classes
        valid = (y_true < nc) & (y_pred < nc) & (y_true >= 0) & (y_pred >= 0)
        if not valid.all():
            y_true = y_true[valid]
            y_pred = y_pred[valid]
        idx = y_true * nc + y_pred
        binc = np.bincount(idx, minlength=nc * nc)
        self.cm += binc.reshape(nc, nc)

    def compute(self):
        if self.cm is None:
            return {"mIoU": 0.0}
        cm = self.cm
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        union = tp + fp + fn
        ious = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
        miou = float(np.nanmean(ious)) if np.isfinite(ious).any() else 0.0
        gt_total = cm.sum(axis=1).astype(np.int64)
        pred_total = cm.sum(axis=0).astype(np.int64)
        return {
            "mIoU": miou,
            "per_class_iou": [None if not np.isfinite(v) else float(v) for v in ious],
            "gt_pixel_count": gt_total.tolist(),
            "pred_pixel_count": pred_total.tolist(),
        }
