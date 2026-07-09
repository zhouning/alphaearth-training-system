from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_patch(path: Path, value: int = 64) -> None:
    rgb = np.full((3, 4, 4), value, dtype=np.uint8)
    np.savez_compressed(path, rgb=rgb)


def _write_source_index(root: Path) -> Path:
    _write_patch(root / "patch_water.npz", value=96)
    np.savez_compressed(root / "arcgis_water.npz", mask=np.full((4, 4), 4, dtype=np.uint8))
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
                "scene_id": "scene_prediction_export",
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


class StubLULCService:
    model_id = "stub-paper12"
    device = "cpu"

    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, ...]] = []

    def predict_image(self, image: np.ndarray) -> dict:
        self.seen_shapes.append(tuple(image.shape))
        return {
            "mask": np.full(image.shape[:2], 2, dtype=np.uint8).tolist(),
            "confidence_summary": {"mean_max_probability": 0.91},
        }


def test_exporter_writes_paper12_masks_without_touching_manual_masks(tmp_path):
    from scripts.export_paper12_packet_predictions import export_packet_predictions

    packet_dir = _build_packet(tmp_path)
    service = StubLULCService()

    summary = export_packet_predictions(
        packet_dir=packet_dir,
        checkpoint_path="fake_checkpoint.pt",
        service_factory=lambda: service,
    )

    output_path = packet_dir / "paper12_masks" / "water_sample.npy"
    assert output_path.exists()
    assert np.load(output_path).tolist() == [[2, 2, 2, 2]] * 4
    assert not list((packet_dir / "manual_masks").glob("*.npy"))
    assert service.seen_shapes == [(4, 4, 3)]

    rows = list(
        csv.DictReader(
            (packet_dir / "arcgis_replacement_annotation_manifest.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    assert rows[0]["paper12_mask_path"] == ""

    assert summary["schema"] == "paper12.arcgis_replacement_packet_prediction_export.v1"
    assert summary["sample_count"] == 1
    assert summary["exported_sample_count"] == 1
    assert summary["skipped_existing_count"] == 0
    assert summary["failed_samples"] == []
    assert summary["outputs"] == [
        {
            "confidence_summary": {"mean_max_probability": 0.91},
            "mask_path": "paper12_masks/water_sample.npy",
            "sample_id": "water_sample",
        }
    ]
    assert (packet_dir / "paper12_prediction_export_summary.json").exists()


def test_exporter_skips_existing_outputs_unless_overwrite(tmp_path):
    from scripts.export_paper12_packet_predictions import export_packet_predictions

    packet_dir = _build_packet(tmp_path)
    output_path = packet_dir / "paper12_masks" / "water_sample.npy"
    np.save(output_path, np.full((4, 4), 5, dtype=np.uint8))
    service = StubLULCService()

    summary = export_packet_predictions(
        packet_dir=packet_dir,
        checkpoint_path="fake_checkpoint.pt",
        service_factory=lambda: service,
    )

    assert np.load(output_path).tolist() == [[5, 5, 5, 5]] * 4
    assert service.seen_shapes == []
    assert summary["exported_sample_count"] == 0
    assert summary["skipped_existing_count"] == 1

    overwrite_summary = export_packet_predictions(
        packet_dir=packet_dir,
        checkpoint_path="fake_checkpoint.pt",
        service_factory=lambda: service,
        overwrite=True,
    )

    assert np.load(output_path).tolist() == [[2, 2, 2, 2]] * 4
    assert service.seen_shapes == [(4, 4, 3)]
    assert overwrite_summary["exported_sample_count"] == 1
    assert overwrite_summary["skipped_existing_count"] == 0


def test_exporter_rejects_prediction_shape_mismatch(tmp_path):
    from scripts.export_paper12_packet_predictions import export_packet_predictions

    class BadShapeService:
        model_id = "bad-shape"
        device = "cpu"

        def predict_image(self, image: np.ndarray) -> dict:
            return {"mask": [[1, 1], [1, 1]]}

    packet_dir = _build_packet(tmp_path)

    summary = export_packet_predictions(
        packet_dir=packet_dir,
        checkpoint_path="fake_checkpoint.pt",
        service_factory=BadShapeService,
    )

    assert summary["exported_sample_count"] == 0
    assert summary["failed_samples"] == [
        {
            "error": "prediction_shape_mismatch",
            "expected_shape": [4, 4],
            "predicted_shape": [2, 2],
            "sample_id": "water_sample",
        }
    ]
    assert not (packet_dir / "paper12_masks" / "water_sample.npy").exists()


def test_default_service_factory_adds_backend_and_repo_roots_to_import_path():
    from scripts.export_paper12_packet_predictions import _default_service_factory

    repo_root = str(REPO_ROOT)
    backend_root = str(REPO_ROOT / "ae_backend")
    original_path = list(sys.path)
    try:
        sys.path[:] = [item for item in sys.path if item not in {repo_root, backend_root}]
        _default_service_factory(
            checkpoint_path="fake_checkpoint.pt",
            prithvi_checkpoint_path=None,
            model_id=None,
            device="cpu",
        )

        assert repo_root in sys.path
        assert backend_root in sys.path
    finally:
        sys.path[:] = original_path


def test_exporter_cli_fails_fast_when_checkpoint_is_missing(tmp_path):
    packet_dir = _build_packet(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/export_paper12_packet_predictions.py"),
            "--packet-dir",
            str(packet_dir),
            "--checkpoint",
            str(tmp_path / "missing_checkpoint.pt"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "Checkpoint not found" in completed.stderr
