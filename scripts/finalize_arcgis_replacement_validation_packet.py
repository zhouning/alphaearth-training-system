"""Finalize a Paper12 ArcGIS-replacement annotation packet.

This script only validates and wires existing evidence. It does not create
manual masks, run Paper12 inference, or alter ArcGIS reference masks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


FINALIZATION_SCHEMA = "paper12.arcgis_replacement_packet_finalization.v1"
ANNOTATION_MANIFEST_NAME = "arcgis_replacement_annotation_manifest.csv"
EVALUATOR_MANIFEST_NAME = "arcgis_replacement_evaluator_manifest.csv"
SUMMARY_NAME = "packet_finalization_summary.json"
EVALUATOR_COLUMNS = [
    "sample_id",
    "manual_mask_path",
    "manual_label",
    "arcgis_mask_path",
    "arcgis_label",
    "paper12_mask_path",
    "paper12_label",
    "scene_id",
    "x",
    "y",
    "dominant_esri_class",
    "dominant_paper12_class",
    "annotator_id",
    "review_status",
]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _resolve(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _relative_to_packet(path: Path, packet_dir: Path) -> str:
    try:
        return path.relative_to(packet_dir).as_posix()
    except ValueError:
        return str(path)


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _default_mask_path(kind: str, sample_id: str) -> str:
    return f"{kind}_masks/{sample_id}.npy"


def _load_mask(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        payload = np.load(path)
        key = "mask" if "mask" in payload.files else sorted(payload.files)[0]
        return np.asarray(payload[key])
    raise ValueError(f"Unsupported mask format for packet finalization: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluator_row(row: dict[str, str], *, manual_path: str, paper12_path: str) -> dict[str, str]:
    return {
        "sample_id": row.get("sample_id", ""),
        "manual_mask_path": manual_path,
        "manual_label": row.get("manual_label", ""),
        "arcgis_mask_path": row.get("arcgis_mask_path", ""),
        "arcgis_label": row.get("arcgis_label", ""),
        "paper12_mask_path": paper12_path,
        "paper12_label": row.get("paper12_label", ""),
        "scene_id": row.get("scene_id", ""),
        "x": row.get("x", ""),
        "y": row.get("y", ""),
        "dominant_esri_class": row.get("dominant_esri_class", ""),
        "dominant_paper12_class": row.get("dominant_paper12_class", ""),
        "annotator_id": row.get("annotator_id", ""),
        "review_status": "ready_for_evaluation",
    }


def finalize_packet(packet_dir: str | Path) -> dict[str, Any]:
    """Create an evaluator manifest once all required packet evidence exists."""
    packet_dir = Path(packet_dir)
    annotation_manifest_path = packet_dir / ANNOTATION_MANIFEST_NAME
    evaluator_manifest_path = packet_dir / EVALUATOR_MANIFEST_NAME
    summary_path = packet_dir / SUMMARY_NAME
    rows = _read_manifest(annotation_manifest_path)

    missing_evidence: list[dict[str, Any]] = []
    shape_errors: list[dict[str, Any]] = []
    evaluator_rows: list[dict[str, str]] = []
    ready_sample_count = 0

    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        manual_path_value = _first_nonempty(
            row.get("manual_mask_path"),
            row.get("manual_mask_target_path"),
            _default_mask_path("manual", sample_id),
        )
        paper12_path_value = _first_nonempty(
            row.get("paper12_mask_path"),
            row.get("paper12_mask_target_path"),
            _default_mask_path("paper12", sample_id),
        )
        arcgis_path_value = _first_nonempty(row.get("arcgis_mask_path"))

        missing: list[str] = []
        manual_path = _resolve(packet_dir, manual_path_value)
        paper12_path = _resolve(packet_dir, paper12_path_value)
        arcgis_path = _resolve(packet_dir, arcgis_path_value) if arcgis_path_value else None
        if not manual_path.exists():
            missing.append("manual_mask")
        if not paper12_path.exists():
            missing.append("paper12_mask")
        if arcgis_path is None or not arcgis_path.exists():
            missing.append("arcgis_mask")

        if missing:
            missing_evidence.append({"missing": missing, "sample_id": sample_id})
            continue

        try:
            arcgis_shape = _load_mask(arcgis_path).shape
            manual_shape = _load_mask(manual_path).shape
            paper12_shape = _load_mask(paper12_path).shape
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
            continue

        ready_sample_count += 1
        evaluator_rows.append(
            _evaluator_row(
                row,
                manual_path=_relative_to_packet(manual_path, packet_dir),
                paper12_path=_relative_to_packet(paper12_path, packet_dir),
            )
        )

    evaluator_ready = (
        bool(rows)
        and ready_sample_count == len(rows)
        and not missing_evidence
        and not shape_errors
    )
    if evaluator_ready:
        with evaluator_manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVALUATOR_COLUMNS)
            writer.writeheader()
            writer.writerows(evaluator_rows)
        evaluator_manifest_value: str | None = str(evaluator_manifest_path)
    else:
        if evaluator_manifest_path.exists():
            evaluator_manifest_path.unlink()
        evaluator_manifest_value = None

    summary: dict[str, Any] = {
        "schema": FINALIZATION_SCHEMA,
        "packet_dir": str(packet_dir),
        "annotation_manifest_path": str(annotation_manifest_path),
        "evaluator_manifest_path": evaluator_manifest_value,
        "sample_count": len(rows),
        "ready_sample_count": ready_sample_count,
        "evaluator_ready": evaluator_ready,
        "missing_evidence": missing_evidence,
        "shape_errors": shape_errors,
    }
    _write_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Finalize a Paper12 ArcGIS replacement validation packet."
    )
    parser.add_argument("--packet-dir", required=True, help="Validation packet directory.")
    args = parser.parse_args(argv)

    print(json.dumps(finalize_packet(args.packet_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
