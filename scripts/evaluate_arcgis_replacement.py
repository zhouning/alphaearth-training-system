"""Evaluate paired ArcGIS-vs-Paper12 replacement evidence.

This script consumes existing paired labels or masks. It does not create manual
truth, run ArcGIS, or run Paper12 inference.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EVALUATION_SCHEMA = "paper12.arcgis_replacement_evaluation.v1"
BOOTSTRAP_SCHEMA = "paper12.arcgis_replacement_bootstrap.v1"
DEFAULT_CLASS_NAMES = [
    "background",
    "built",
    "crops",
    "trees",
    "water",
    "rangeland_bare",
]
DEFAULT_CRITICAL_CLASSES = ["water", "crops", "built"]
DEFAULT_MIN_CANDIDATE_ROWS = 30


def _flatten_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ignore_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if truth.shape != pred.shape:
        raise ValueError(
            f"Mask shapes differ: {np.asarray(y_true).shape} vs "
            f"{np.asarray(y_pred).shape}"
        )
    if ignore_index is not None:
        keep = truth != ignore_index
        truth = truth[keep]
        pred = pred[keep]
    return truth.astype(np.int64), pred.astype(np.int64)


def _confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    valid = (
        (y_true >= 0)
        & (y_true < num_classes)
        & (y_pred >= 0)
        & (y_pred < num_classes)
    )
    truth = y_true[valid]
    pred = y_pred[valid]
    encoded = truth * num_classes + pred
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).astype(np.int64)


def _metrics_from_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: Sequence[str],
) -> dict[str, object]:
    total = int(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    union = tp + fp + fn
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)

    f1_denominator = (2 * tp) + fp + fn
    f1 = np.divide(
        2 * tp,
        f1_denominator,
        out=np.full_like(tp, np.nan),
        where=f1_denominator > 0,
    )

    finite_iou = iou[np.isfinite(iou)]
    finite_f1 = f1[np.isfinite(f1)]

    per_class_iou = {
        name: (None if not np.isfinite(value) else float(value))
        for name, value in zip(class_names, iou)
    }
    return {
        "overall_accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(finite_f1.mean()) if finite_f1.size else 0.0,
        "per_class_iou": per_class_iou,
        "miou": float(finite_iou.mean()) if finite_iou.size else 0.0,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def compute_replacement_metrics(
    *,
    manual: np.ndarray,
    arcgis: np.ndarray,
    paper12: np.ndarray,
    class_names: Sequence[str],
    ignore_index: int | None = 255,
) -> dict[str, object]:
    """Compare ArcGIS and Paper12 predictions against the same manual mask."""
    if not class_names:
        raise ValueError("class_names must contain at least one class")

    num_classes = len(class_names)
    manual_arcgis, arcgis_flat = _flatten_pair(
        manual,
        arcgis,
        ignore_index=ignore_index,
    )
    manual_paper12, paper12_flat = _flatten_pair(
        manual,
        paper12,
        ignore_index=ignore_index,
    )

    arcgis_metrics = _metrics_from_confusion_matrix(
        _confusion_matrix(
            manual_arcgis,
            arcgis_flat,
            num_classes=num_classes,
        ),
        class_names=class_names,
    )
    paper12_metrics = _metrics_from_confusion_matrix(
        _confusion_matrix(
            manual_paper12,
            paper12_flat,
            num_classes=num_classes,
        ),
        class_names=class_names,
    )

    return {
        "schema": EVALUATION_SCHEMA,
        "pixel_count": int(manual_arcgis.size),
        "class_names": list(class_names),
        "arcgis_vs_manual": arcgis_metrics,
        "paper12_vs_manual": paper12_metrics,
        "paired_delta": {
            "overall_accuracy": float(
                paper12_metrics["overall_accuracy"]
                - arcgis_metrics["overall_accuracy"]
            ),
            "macro_f1": float(
                paper12_metrics["macro_f1"] - arcgis_metrics["macro_f1"]
            ),
            "miou": float(paper12_metrics["miou"] - arcgis_metrics["miou"]),
        },
    }


def _bootstrap_paired_delta_intervals(
    *,
    manual_parts: Sequence[np.ndarray],
    arcgis_parts: Sequence[np.ndarray],
    paper12_parts: Sequence[np.ndarray],
    class_names: Sequence[str],
    ignore_index: int | None,
    point_delta: dict[str, float],
    iterations: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive when enabled")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if not manual_parts:
        raise ValueError("Cannot bootstrap an empty manifest")

    rng = np.random.default_rng(seed)
    row_count = len(manual_parts)
    alpha = (1.0 - confidence_level) / 2.0
    quantiles = [alpha, 1.0 - alpha]
    deltas: dict[str, list[float]] = {
        "overall_accuracy": [],
        "macro_f1": [],
        "miou": [],
    }

    for _ in range(iterations):
        indices = rng.integers(0, row_count, size=row_count)
        metrics = compute_replacement_metrics(
            manual=np.concatenate([manual_parts[int(index)] for index in indices]),
            arcgis=np.concatenate([arcgis_parts[int(index)] for index in indices]),
            paper12=np.concatenate([paper12_parts[int(index)] for index in indices]),
            class_names=class_names,
            ignore_index=ignore_index,
        )
        for metric_name in deltas:
            deltas[metric_name].append(float(metrics["paired_delta"][metric_name]))

    paired_delta_ci: dict[str, dict[str, float | str]] = {}
    for metric_name, values in deltas.items():
        sample = np.asarray(values, dtype=np.float64)
        lower, upper = np.quantile(sample, quantiles)
        paired_delta_ci[metric_name] = {
            "metric": metric_name,
            "point_estimate": float(point_delta[metric_name]),
            "mean": float(sample.mean()),
            "lower": float(lower),
            "upper": float(upper),
        }

    return {
        "schema": BOOTSTRAP_SCHEMA,
        "sample_unit": "manifest_row",
        "iterations": int(iterations),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "row_count": int(row_count),
        "paired_delta_ci": paired_delta_ci,
    }

def decide_replacement_status(
    metrics: dict[str, Any] | None,
    *,
    critical_classes: Sequence[str] = DEFAULT_CRITICAL_CLASSES,
    tolerance: float = 0.0,
) -> dict[str, object]:
    """Apply the conservative ArcGIS replacement decision rule."""
    if metrics is None:
        return {
            "decision_status": "not_validated",
            "replacement_claim_supported": False,
            "arcgis_replacement_ready": False,
            "reasons": ["missing_paired_evidence"],
        }

    arcgis_metrics = metrics["arcgis_vs_manual"]
    paper12_metrics = metrics["paper12_vs_manual"]
    reasons: list[str] = []

    arcgis_miou = float(arcgis_metrics["miou"])
    paper12_miou = float(paper12_metrics["miou"])
    if paper12_miou < arcgis_miou - tolerance:
        reasons.append("paper12_below_arcgis_miou")

    arcgis_iou = arcgis_metrics.get("per_class_iou", {})
    paper12_iou = paper12_metrics.get("per_class_iou", {})
    for class_name in critical_classes:
        arcgis_value = arcgis_iou.get(class_name)
        paper12_value = paper12_iou.get(class_name)
        if arcgis_value is None or paper12_value is None:
            reasons.append(f"missing_critical_class:{class_name}")
            continue
        if float(paper12_value) < float(arcgis_value) - tolerance:
            reasons.append(f"paper12_below_arcgis_critical_class:{class_name}")

    status = "partial" if reasons else "replacement_candidate"
    supported = status == "replacement_candidate"
    return {
        "decision_status": status,
        "replacement_claim_supported": supported,
        "arcgis_replacement_ready": supported,
        "reasons": reasons,
    }


def _apply_minimum_sample_guard(
    decision: dict[str, object],
    *,
    manifest_row_count: int,
    min_candidate_rows: int,
) -> dict[str, object]:
    if min_candidate_rows < 1:
        raise ValueError("min_candidate_rows must be at least 1")
    if decision["decision_status"] != "replacement_candidate":
        return decision
    if manifest_row_count >= min_candidate_rows:
        return decision

    reasons = list(decision.get("reasons", []))
    reasons.append(f"insufficient_manifest_rows:{manifest_row_count}<{min_candidate_rows}")
    return {
        "decision_status": "insufficient_sample_size",
        "replacement_claim_supported": False,
        "arcgis_replacement_ready": False,
        "reasons": reasons,
    }


def _nonempty(value: str | None) -> str:
    return (value or "").strip()


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _load_mask(mask_path: Path) -> np.ndarray:
    suffix = mask_path.suffix.lower()
    if suffix == ".npy":
        return np.load(mask_path)
    if suffix == ".npz":
        payload = np.load(mask_path)
        key = "mask" if "mask" in payload.files else sorted(payload.files)[0]
        return payload[key]
    if suffix == ".csv":
        return np.loadtxt(mask_path, delimiter=",", dtype=np.int64)

    try:
        import rasterio  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ValueError(
            f"Unsupported mask format without rasterio: {mask_path}"
        ) from exc

    with rasterio.open(mask_path) as dataset:  # pragma: no cover - optional package
        return dataset.read(1)


def _load_evidence(
    row: dict[str, str],
    *,
    base_dir: Path,
    mask_column: str,
    label_column: str,
) -> np.ndarray | None:
    mask_value = _nonempty(row.get(mask_column))
    label_value = _nonempty(row.get(label_column))
    if mask_value:
        return np.asarray(_load_mask(_resolve_path(base_dir, mask_value)))
    if label_value:
        return np.asarray([int(label_value)], dtype=np.int64)
    return None


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _missing_evidence(rows: Sequence[dict[str, str]]) -> list[str]:
    missing: set[str] = set()
    if not rows:
        return ["manifest_rows"]

    checks = [
        ("manual_ground_truth", "manual_mask_path", "manual_label"),
        ("arcgis_reference_output", "arcgis_mask_path", "arcgis_label"),
        ("paper12_output", "paper12_mask_path", "paper12_label"),
    ]
    for row in rows:
        for evidence_name, mask_column, label_column in checks:
            if not (
                _nonempty(row.get(mask_column))
                or _nonempty(row.get(label_column))
            ):
                missing.add(evidence_name)
    return sorted(missing)


def evaluate_manifest(
    manifest_path: str | Path,
    *,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    critical_classes: Sequence[str] = DEFAULT_CRITICAL_CLASSES,
    tolerance: float = 0.0,
    ignore_index: int | None = 255,
    bootstrap_iterations: int = 0,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
    min_candidate_rows: int = DEFAULT_MIN_CANDIDATE_ROWS,
) -> dict[str, object]:
    """Evaluate a manifest of paired manual, ArcGIS, and Paper12 outputs."""
    manifest_path = Path(manifest_path)
    rows = _read_manifest(manifest_path)
    missing = _missing_evidence(rows)
    base_payload: dict[str, object] = {
        "schema": EVALUATION_SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_row_count": len(rows),
        "class_names": list(class_names),
        "critical_classes": list(critical_classes),
        "tolerance": float(tolerance),
        "min_candidate_rows": int(min_candidate_rows),
    }

    if missing:
        decision = decide_replacement_status(None)
        return {
            **base_payload,
            "metrics": None,
            "bootstrap": None,
            "missing_evidence": missing,
            **decision,
        }

    manual_parts: list[np.ndarray] = []
    arcgis_parts: list[np.ndarray] = []
    paper12_parts: list[np.ndarray] = []
    for row in rows:
        manual = _load_evidence(
            row,
            base_dir=manifest_path.parent,
            mask_column="manual_mask_path",
            label_column="manual_label",
        )
        arcgis = _load_evidence(
            row,
            base_dir=manifest_path.parent,
            mask_column="arcgis_mask_path",
            label_column="arcgis_label",
        )
        paper12 = _load_evidence(
            row,
            base_dir=manifest_path.parent,
            mask_column="paper12_mask_path",
            label_column="paper12_label",
        )
        if manual is None or arcgis is None or paper12 is None:
            raise ValueError("Manifest evidence changed after validation")

        manual_parts.append(np.asarray(manual).ravel())
        arcgis_parts.append(np.asarray(arcgis).ravel())
        paper12_parts.append(np.asarray(paper12).ravel())

    metrics = compute_replacement_metrics(
        manual=np.concatenate(manual_parts),
        arcgis=np.concatenate(arcgis_parts),
        paper12=np.concatenate(paper12_parts),
        class_names=class_names,
        ignore_index=ignore_index,
    )
    decision = decide_replacement_status(
        metrics,
        critical_classes=critical_classes,
        tolerance=tolerance,
    )
    decision = _apply_minimum_sample_guard(
        decision,
        manifest_row_count=len(rows),
        min_candidate_rows=min_candidate_rows,
    )
    bootstrap = None
    if bootstrap_iterations:
        bootstrap = _bootstrap_paired_delta_intervals(
            manual_parts=manual_parts,
            arcgis_parts=arcgis_parts,
            paper12_parts=paper12_parts,
            class_names=class_names,
            ignore_index=ignore_index,
            point_delta=metrics["paired_delta"],
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    return {
        **base_payload,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "missing_evidence": [],
        **decision,
    }


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate paired ArcGIS and Paper12 outputs against manual labels."
    )
    parser.add_argument("--manifest", required=True, help="Validation manifest CSV.")
    parser.add_argument("--output", help="Destination JSON file.")
    parser.add_argument(
        "--class-names",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names in integer-id order.",
    )
    parser.add_argument(
        "--critical-classes",
        default=",".join(DEFAULT_CRITICAL_CLASSES),
        help="Comma-separated classes that must match ArcGIS within tolerance.",
    )
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=0,
        help="Optional row-level bootstrap iterations for paired delta intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed used when --bootstrap-iterations is enabled.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for optional bootstrap intervals.",
    )
    parser.add_argument(
        "--min-candidate-rows",
        type=int,
        default=DEFAULT_MIN_CANDIDATE_ROWS,
        help="Minimum manifest rows required before reporting replacement_candidate.",
    )
    args = parser.parse_args(argv)

    payload = evaluate_manifest(
        args.manifest,
        class_names=_parse_csv_list(args.class_names),
        critical_classes=_parse_csv_list(args.critical_classes),
        tolerance=args.tolerance,
        ignore_index=args.ignore_index,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        min_candidate_rows=args.min_candidate_rows,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()