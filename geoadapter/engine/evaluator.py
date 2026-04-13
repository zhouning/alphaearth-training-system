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


def compute_segmentation_metrics(y_true, y_pred):
    """Compute mean IoU for semantic segmentation."""
    classes = np.unique(np.concatenate([y_true.flatten(), y_pred.flatten()]))
    ious = []
    for c in classes:
        intersection = np.sum((y_true == c) & (y_pred == c))
        union = np.sum((y_true == c) | (y_pred == c))
        if union > 0:
            ious.append(intersection / union)
    return {"mIoU": np.mean(ious) if ious else 0.0}
