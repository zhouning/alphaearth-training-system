from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_patch(path: Path, value: int = 64) -> None:
    rgb = np.full((3, 4, 4), value, dtype=np.uint8)
    np.savez_compressed(path, rgb=rgb)


def _write_mask(path: Path, class_id: int) -> None:
    mask = np.full((4, 4), class_id, dtype=np.uint8)
    np.save(path, mask)


def _critical_class_mask() -> np.ndarray:
    return np.array(
        [
            [4, 4, 4, 4],
            [4, 4, 1, 1],
            [2, 2, 4, 4],
            [1, 1, 2, 2],
        ],
        dtype=np.uint8,
    )


def _write_critical_class_mask(path: Path) -> None:
    np.save(path, _critical_class_mask())


def _write_source_index(root: Path) -> Path:
    _write_patch(root / "patch_water.npz", value=96)
    np.savez_compressed(root / "arcgis_water.npz", mask=_critical_class_mask())
    index_path = root / "source_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "scene_id", "year", "patch_path", "lulc_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "water_sample",
                "scene_id": "scene_finalizer",
                "year": "2022",
                "patch_path": "patch_water.npz",
                "lulc_path": "arcgis_water.npz",
            }
        )
    return index_path


def _build_packet(root: Path) -> Path:
    from scripts.prepare_arcgis_replacement_validation_packet import build_validation_packet

    packet_dir = root / "packet"
    build_validation_packet(
        index_path=_write_source_index(root),
        output_dir=packet_dir,
        sample_count=1,
        year=2022,
        required_classes=["water"],
    )
    return packet_dir


def test_packet_finalizer_reports_missing_manual_and_paper12_outputs(tmp_path):
    from scripts.finalize_arcgis_replacement_validation_packet import finalize_packet

    packet_dir = _build_packet(tmp_path)

    summary = finalize_packet(packet_dir=packet_dir)

    assert summary["schema"] == "paper12.arcgis_replacement_packet_finalization.v1"
    assert summary["evaluator_ready"] is False
    assert summary["sample_count"] == 1
    assert summary["ready_sample_count"] == 0
    assert summary["evaluator_manifest_path"] is None
    assert summary["missing_evidence"] == [
        {
            "missing": ["manual_mask", "paper12_mask"],
            "sample_id": "water_sample",
        }
    ]
    assert not (packet_dir / "arcgis_replacement_evaluator_manifest.csv").exists()
    assert (packet_dir / "packet_finalization_summary.json").exists()


def test_packet_finalizer_writes_evaluator_manifest_when_outputs_exist(tmp_path):
    from scripts.evaluate_arcgis_replacement import evaluate_manifest
    from scripts.finalize_arcgis_replacement_validation_packet import finalize_packet

    packet_dir = _build_packet(tmp_path)
    _write_critical_class_mask(packet_dir / "manual_masks" / "water_sample.npy")
    _write_critical_class_mask(packet_dir / "paper12_masks" / "water_sample.npy")

    summary = finalize_packet(packet_dir=packet_dir)

    assert summary["evaluator_ready"] is True
    assert summary["ready_sample_count"] == 1
    assert summary["missing_evidence"] == []
    manifest_path = Path(summary["evaluator_manifest_path"])
    assert manifest_path == packet_dir / "arcgis_replacement_evaluator_manifest.csv"

    rows = list(csv.DictReader(manifest_path.read_text(encoding="utf-8").splitlines()))
    assert rows == [
        {
            "sample_id": "water_sample",
            "manual_mask_path": "manual_masks/water_sample.npy",
            "manual_label": "",
            "arcgis_mask_path": "arcgis_masks/water_sample.npy",
            "arcgis_label": "",
            "paper12_mask_path": "paper12_masks/water_sample.npy",
            "paper12_label": "",
            "scene_id": "scene_finalizer",
            "x": "",
            "y": "",
            "dominant_esri_class": "water",
            "dominant_paper12_class": "",
            "annotator_id": "",
            "review_status": "ready_for_evaluation",
        }
    ]

    class_names = ["background", "built", "crops", "trees", "water", "rangeland_bare"]
    evaluation = evaluate_manifest(manifest_path, class_names=class_names)
    assert evaluation["decision_status"] == "insufficient_sample_size"
    assert evaluation["replacement_claim_supported"] is False
    assert "insufficient_manifest_rows:1<30" in evaluation["reasons"]

    smoke_evaluation = evaluate_manifest(
        manifest_path,
        class_names=class_names,
        min_candidate_rows=1,
        min_critical_class_rows=1,
    )
    assert smoke_evaluation["decision_status"] == "replacement_candidate"
    assert smoke_evaluation["replacement_claim_supported"] is True


def test_packet_finalizer_records_replacement_candidate_sample_size_gap(tmp_path):
    from scripts.finalize_arcgis_replacement_validation_packet import finalize_packet

    packet_dir = _build_packet(tmp_path)
    _write_critical_class_mask(packet_dir / "manual_masks" / "water_sample.npy")
    _write_critical_class_mask(packet_dir / "paper12_masks" / "water_sample.npy")

    summary = finalize_packet(packet_dir=packet_dir)

    assert summary["min_candidate_rows"] == 30
    assert summary["replacement_candidate_sample_size_ready"] is False
    assert summary["replacement_candidate_sample_size_gap"] == 29
    assert summary["evaluator_ready"] is True

    smoke_summary = finalize_packet(packet_dir=packet_dir, min_candidate_rows=1)
    assert smoke_summary["replacement_candidate_sample_size_ready"] is True
    assert smoke_summary["replacement_candidate_sample_size_gap"] == 0


def test_packet_finalizer_rejects_invalid_min_candidate_rows(tmp_path):
    from scripts.finalize_arcgis_replacement_validation_packet import finalize_packet

    packet_dir = _build_packet(tmp_path)

    with pytest.raises(ValueError, match="min_candidate_rows"):
        finalize_packet(packet_dir=packet_dir, min_candidate_rows=0)


def test_packet_finalizer_rejects_shape_mismatched_outputs(tmp_path):
    from scripts.finalize_arcgis_replacement_validation_packet import finalize_packet

    packet_dir = _build_packet(tmp_path)
    np.save(packet_dir / "manual_masks" / "water_sample.npy", np.full((5, 5), 4, dtype=np.uint8))
    _write_critical_class_mask(packet_dir / "paper12_masks" / "water_sample.npy")

    summary = finalize_packet(packet_dir=packet_dir)

    assert summary["evaluator_ready"] is False
    assert summary["ready_sample_count"] == 0
    assert summary["missing_evidence"] == []
    assert summary["shape_errors"] == [
        {
            "arcgis_shape": [4, 4],
            "manual_shape": [5, 5],
            "paper12_shape": [4, 4],
            "sample_id": "water_sample",
        }
    ]
    assert not (packet_dir / "arcgis_replacement_evaluator_manifest.csv").exists()


def test_packet_finalizer_cli_writes_summary(tmp_path):
    packet_dir = _build_packet(tmp_path)
    _write_critical_class_mask(packet_dir / "manual_masks" / "water_sample.npy")
    _write_critical_class_mask(packet_dir / "paper12_masks" / "water_sample.npy")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/finalize_arcgis_replacement_validation_packet.py"),
            "--packet-dir",
            str(packet_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(
        (packet_dir / "packet_finalization_summary.json").read_text(encoding="utf-8")
    )
    assert summary["evaluator_ready"] is True
    assert Path(summary["evaluator_manifest_path"]).exists()
