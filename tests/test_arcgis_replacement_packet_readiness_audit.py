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


def _write_mask(path: Path, mask: np.ndarray | None = None) -> None:
    np.save(path, _critical_class_mask() if mask is None else mask)


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
                "scene_id": "scene_readiness",
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


def test_readiness_audit_reports_missing_manual_and_paper12_outputs(tmp_path):
    from scripts.audit_arcgis_replacement_validation_packet import audit_packet_readiness

    packet_dir = _build_packet(tmp_path)

    summary = audit_packet_readiness(
        packet_dir=packet_dir,
        checkpoint_path="data/weights/linhe_lulc/houlsby__rgb_3band__seed123.pt",
    )

    assert summary["schema"] == "paper12.arcgis_replacement_packet_readiness.v1"
    assert summary["status"] == "waiting_for_manual_and_paper12"
    assert summary["sample_count"] == 1
    assert summary["available_counts"] == {
        "arcgis_mask": 1,
        "manual_mask": 0,
        "paper12_mask": 0,
    }
    assert summary["evidence_ready_for_finalizer"] is False
    assert summary["evaluator_manifest_ready"] is False
    assert summary["missing_evidence_by_sample"] == [
        {
            "missing": ["manual_mask", "paper12_mask"],
            "sample_id": "water_sample",
        }
    ]
    assert summary["shape_errors"] == []
    assert any("manual_masks/<sample_id>.npy" in action for action in summary["next_actions"])
    assert any("scripts/export_paper12_packet_predictions.py" in action for action in summary["next_actions"])
    assert (packet_dir / "packet_readiness_summary.json").exists()


def test_readiness_audit_reports_replacement_candidate_sample_size_gap(tmp_path):
    from scripts.audit_arcgis_replacement_validation_packet import audit_packet_readiness

    packet_dir = _build_packet(tmp_path)

    summary = audit_packet_readiness(packet_dir=packet_dir)

    assert summary["min_candidate_rows"] == 30
    assert summary["replacement_candidate_sample_size_ready"] is False
    assert summary["replacement_candidate_sample_size_gap"] == 29
    assert any(
        "29 more validation samples" in action for action in summary["next_actions"]
    )

    smoke_summary = audit_packet_readiness(packet_dir=packet_dir, min_candidate_rows=1)
    assert smoke_summary["replacement_candidate_sample_size_ready"] is True
    assert smoke_summary["replacement_candidate_sample_size_gap"] == 0


def test_readiness_audit_rejects_invalid_min_candidate_rows(tmp_path):
    from scripts.audit_arcgis_replacement_validation_packet import audit_packet_readiness

    packet_dir = _build_packet(tmp_path)

    with pytest.raises(ValueError, match="min_candidate_rows"):
        audit_packet_readiness(packet_dir=packet_dir, min_candidate_rows=0)


def test_readiness_audit_reports_only_missing_manual_after_paper12_export(tmp_path):
    from scripts.audit_arcgis_replacement_validation_packet import audit_packet_readiness

    packet_dir = _build_packet(tmp_path)
    _write_mask(packet_dir / "paper12_masks" / "water_sample.npy")

    summary = audit_packet_readiness(packet_dir=packet_dir)

    assert summary["status"] == "waiting_for_manual_ground_truth"
    assert summary["available_counts"] == {
        "arcgis_mask": 1,
        "manual_mask": 0,
        "paper12_mask": 1,
    }
    assert summary["missing_evidence_by_sample"] == [
        {
            "missing": ["manual_mask"],
            "sample_id": "water_sample",
        }
    ]
    assert any("manual_masks/<sample_id>.npy" in action for action in summary["next_actions"])
    assert not any("export_paper12_packet_predictions.py" in action for action in summary["next_actions"])


def test_readiness_audit_advances_from_finalizer_ready_to_evaluator_ready(tmp_path):
    from scripts.audit_arcgis_replacement_validation_packet import audit_packet_readiness
    from scripts.finalize_arcgis_replacement_validation_packet import finalize_packet

    packet_dir = _build_packet(tmp_path)
    _write_mask(packet_dir / "manual_masks" / "water_sample.npy")
    _write_mask(packet_dir / "paper12_masks" / "water_sample.npy")

    before_finalizer = audit_packet_readiness(packet_dir=packet_dir)

    assert before_finalizer["status"] == "ready_for_finalization"
    assert before_finalizer["evidence_ready_for_finalizer"] is True
    assert before_finalizer["evaluator_manifest_ready"] is False
    assert any(
        "scripts/finalize_arcgis_replacement_validation_packet.py" in action
        for action in before_finalizer["next_actions"]
    )

    finalize_packet(packet_dir=packet_dir)
    after_finalizer = audit_packet_readiness(packet_dir=packet_dir)

    assert after_finalizer["status"] == "ready_for_evaluation"
    assert after_finalizer["evidence_ready_for_finalizer"] is True
    assert after_finalizer["evaluator_manifest_ready"] is True
    assert any(
        "scripts/evaluate_arcgis_replacement.py" in action
        for action in after_finalizer["next_actions"]
    )


def test_readiness_audit_blocks_shape_mismatches(tmp_path):
    from scripts.audit_arcgis_replacement_validation_packet import audit_packet_readiness

    packet_dir = _build_packet(tmp_path)
    _write_mask(packet_dir / "manual_masks" / "water_sample.npy")
    _write_mask(
        packet_dir / "paper12_masks" / "water_sample.npy",
        np.full((2, 2), 4, dtype=np.uint8),
    )

    summary = audit_packet_readiness(packet_dir=packet_dir)

    assert summary["status"] == "blocked_by_shape_errors"
    assert summary["evidence_ready_for_finalizer"] is False
    assert summary["shape_errors"] == [
        {
            "arcgis_shape": [4, 4],
            "manual_shape": [4, 4],
            "paper12_shape": [2, 2],
            "sample_id": "water_sample",
        }
    ]
    assert any("Fix mask shape mismatches" in action for action in summary["next_actions"])


def test_readiness_audit_cli_writes_summary(tmp_path):
    packet_dir = _build_packet(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/audit_arcgis_replacement_validation_packet.py"),
            "--packet-dir",
            str(packet_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((packet_dir / "packet_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "waiting_for_manual_and_paper12"
