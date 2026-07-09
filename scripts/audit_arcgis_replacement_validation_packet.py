"""Audit readiness of a Paper12 ArcGIS-replacement validation packet.

The audit is diagnostic only: it does not create manual labels, run inference,
finalize manifests, or evaluate replacement status.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


READINESS_SCHEMA = "paper12.arcgis_replacement_packet_readiness.v1"
ANNOTATION_MANIFEST_NAME = "arcgis_replacement_annotation_manifest.csv"
EVALUATOR_MANIFEST_NAME = "arcgis_replacement_evaluator_manifest.csv"
SUMMARY_NAME = "packet_readiness_summary.json"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _resolve(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _default_mask_path(kind: str, sample_id: str) -> str:
    return f"{kind}_masks/{sample_id}.npy"


def _load_mask_shape(path: Path) -> tuple[int, ...]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return tuple(np.load(path).shape)
    if suffix == ".npz":
        payload = np.load(path)
        key = "mask" if "mask" in payload.files else sorted(payload.files)[0]
        return tuple(payload[key].shape)
    raise ValueError(f"Unsupported mask format for readiness audit: {path}")


def _paths_for_row(packet_dir: Path, row: dict[str, str]) -> dict[str, Path | None]:
    sample_id = row.get("sample_id", "").strip()
    manual_value = _first_nonempty(
        row.get("manual_mask_path"),
        row.get("manual_mask_target_path"),
        _default_mask_path("manual", sample_id),
    )
    paper12_value = _first_nonempty(
        row.get("paper12_mask_path"),
        row.get("paper12_mask_target_path"),
        _default_mask_path("paper12", sample_id),
    )
    arcgis_value = _first_nonempty(
        row.get("arcgis_mask_path"),
        _default_mask_path("arcgis", sample_id),
    )
    return {
        "manual_mask": _resolve(packet_dir, manual_value),
        "paper12_mask": _resolve(packet_dir, paper12_value),
        "arcgis_mask": _resolve(packet_dir, arcgis_value) if arcgis_value else None,
    }


def _status(
    *,
    rows: Sequence[dict[str, str]],
    missing_evidence_by_sample: Sequence[dict[str, Any]],
    shape_errors: Sequence[dict[str, Any]],
    evaluator_manifest_ready: bool,
) -> str:
    if not rows:
        return "empty_packet"
    if shape_errors:
        return "blocked_by_shape_errors"
    if not missing_evidence_by_sample:
        return "ready_for_evaluation" if evaluator_manifest_ready else "ready_for_finalization"

    missing_names = {
        missing_name
        for item in missing_evidence_by_sample
        for missing_name in item["missing"]
    }
    if missing_names == {"manual_mask"}:
        return "waiting_for_manual_ground_truth"
    if missing_names == {"paper12_mask"}:
        return "waiting_for_paper12_predictions"
    if missing_names == {"manual_mask", "paper12_mask"}:
        return "waiting_for_manual_and_paper12"
    if "arcgis_mask" in missing_names:
        return "missing_arcgis_reference"
    return "waiting_for_evidence"


def _next_actions(
    *,
    status: str,
    packet_dir: Path,
    checkpoint_path: str | Path | None,
) -> list[str]:
    packet_arg = str(packet_dir)
    checkpoint = str(checkpoint_path or "<paper12_checkpoint.pt>")
    if status == "empty_packet":
        return ["Rebuild the packet with at least one validation sample."]
    if status == "blocked_by_shape_errors":
        return ["Fix mask shape mismatches before running the packet finalizer."]
    if status == "ready_for_finalization":
        return [
            "Run scripts/finalize_arcgis_replacement_validation_packet.py "
            f"--packet-dir {packet_arg}."
        ]
    if status == "ready_for_evaluation":
        return [
            "Run scripts/evaluate_arcgis_replacement.py --manifest "
            f"{packet_dir / EVALUATOR_MANIFEST_NAME}."
        ]
    actions: list[str] = []
    if status in {"waiting_for_manual_ground_truth", "waiting_for_manual_and_paper12", "missing_arcgis_reference", "waiting_for_evidence"}:
        actions.append(
            "Save independent manual masks to manual_masks/<sample_id>.npy for "
            "every packet sample."
        )
    if status in {"waiting_for_paper12_predictions", "waiting_for_manual_and_paper12", "waiting_for_evidence"}:
        actions.append(
            "Run scripts/export_paper12_packet_predictions.py "
            f"--packet-dir {packet_arg} --checkpoint {checkpoint}."
        )
    if status == "missing_arcgis_reference":
        actions.append("Regenerate the packet so arcgis_masks/<sample_id>.npy exists.")
    return actions


def audit_packet_readiness(
    *,
    packet_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect whether a validation packet can be finalized or evaluated."""
    packet_dir = Path(packet_dir)
    annotation_manifest_path = packet_dir / ANNOTATION_MANIFEST_NAME
    evaluator_manifest_path = packet_dir / EVALUATOR_MANIFEST_NAME
    output_path = Path(output_path) if output_path is not None else packet_dir / SUMMARY_NAME
    rows = _read_manifest(annotation_manifest_path)

    available_counts = {"manual_mask": 0, "paper12_mask": 0, "arcgis_mask": 0}
    missing_evidence_by_sample: list[dict[str, Any]] = []
    shape_errors: list[dict[str, Any]] = []

    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        paths = _paths_for_row(packet_dir, row)
        missing: list[str] = []
        for evidence_name, path in paths.items():
            if path is None or not path.exists():
                missing.append(evidence_name)
            else:
                available_counts[evidence_name] += 1

        if missing:
            missing_evidence_by_sample.append(
                {"missing": missing, "sample_id": sample_id}
            )
            continue

        try:
            arcgis_shape = _load_mask_shape(paths["arcgis_mask"])  # type: ignore[arg-type]
            manual_shape = _load_mask_shape(paths["manual_mask"])  # type: ignore[arg-type]
            paper12_shape = _load_mask_shape(paths["paper12_mask"])  # type: ignore[arg-type]
        except ValueError as exc:
            shape_errors.append({"error": str(exc), "sample_id": sample_id})
            continue

        if manual_shape != arcgis_shape or paper12_shape != arcgis_shape:
            shape_errors.append(
                {
                    "arcgis_shape": list(arcgis_shape),
                    "manual_shape": list(manual_shape),
                    "paper12_shape": list(paper12_shape),
                    "sample_id": sample_id,
                }
            )

    evaluator_manifest_ready = evaluator_manifest_path.exists()
    evidence_ready = (
        bool(rows)
        and not missing_evidence_by_sample
        and not shape_errors
    )
    status = _status(
        rows=rows,
        missing_evidence_by_sample=missing_evidence_by_sample,
        shape_errors=shape_errors,
        evaluator_manifest_ready=evaluator_manifest_ready,
    )
    summary: dict[str, Any] = {
        "schema": READINESS_SCHEMA,
        "packet_dir": str(packet_dir),
        "annotation_manifest_path": str(annotation_manifest_path),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "sample_count": len(rows),
        "available_counts": available_counts,
        "evidence_ready_for_finalizer": evidence_ready,
        "evaluator_manifest_path": str(evaluator_manifest_path),
        "evaluator_manifest_ready": evaluator_manifest_ready,
        "status": status,
        "missing_evidence_by_sample": missing_evidence_by_sample,
        "shape_errors": shape_errors,
        "next_actions": _next_actions(
            status=status,
            packet_dir=packet_dir,
            checkpoint_path=checkpoint_path,
        ),
    }
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit readiness of a Paper12 ArcGIS validation packet."
    )
    parser.add_argument("--packet-dir", required=True, help="Validation packet directory.")
    parser.add_argument("--checkpoint", help="Optional Paper12 checkpoint for next-action text.")
    parser.add_argument("--output", help="Optional readiness summary JSON path.")
    args = parser.parse_args(argv)

    try:
        summary = audit_packet_readiness(
            packet_dir=args.packet_dir,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
        )
    except Exception as exc:
        print(str(exc))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
