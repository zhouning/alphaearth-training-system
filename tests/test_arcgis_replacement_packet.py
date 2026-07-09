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


def _write_mask(path: Path, class_id: int) -> None:
    mask = np.full((4, 4), class_id, dtype=np.uint8)
    np.savez_compressed(path, mask=mask)


def _write_index(root: Path) -> Path:
    rows = [
        ("s_water", "scene_a", 2022, "patch_water.npz", "mask_water.npz"),
        ("s_crops", "scene_b", 2022, "patch_crops.npz", "mask_crops.npz"),
        ("s_built", "scene_c", 2022, "patch_built.npz", "mask_built.npz"),
        ("s_trees", "scene_d", 2022, "patch_trees.npz", "mask_trees.npz"),
    ]
    for _, _, _, patch_name, mask_name in rows:
        _write_patch(root / patch_name)
        class_id = {
            "mask_water.npz": 4,
            "mask_crops.npz": 2,
            "mask_built.npz": 1,
            "mask_trees.npz": 3,
        }[mask_name]
        _write_mask(root / mask_name, class_id)

    index_path = root / "lulc_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "scene_id",
                "year",
                "patch_path",
                "lulc_path",
                "minx",
                "miny",
                "maxx",
                "maxy",
            ],
        )
        writer.writeheader()
        for sample_id, scene_id, year, patch_name, mask_name in rows:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "year": year,
                    "patch_path": patch_name,
                    "lulc_path": mask_name,
                    "minx": "0",
                    "miny": "1",
                    "maxx": "2",
                    "maxy": "3",
                }
            )
    return index_path


def test_packet_builder_creates_conservative_annotation_packet(tmp_path):
    from scripts.prepare_arcgis_replacement_validation_packet import (
        build_validation_packet,
    )

    index_path = _write_index(tmp_path)
    output_dir = tmp_path / "packet"

    summary = build_validation_packet(
        index_path=index_path,
        output_dir=output_dir,
        sample_count=3,
        year=2022,
        seed=7,
        required_classes=["water", "crops", "built"],
    )

    assert summary["schema"] == "paper12.arcgis_replacement_validation_packet.v1"
    assert summary["sample_count"] == 3
    assert summary["evaluator_ready"] is False
    assert summary["manual_ground_truth_available"] is False
    assert summary["paper12_outputs_available"] is False
    assert summary["critical_classes_requested"] == ["water", "crops", "built"]
    assert summary["critical_classes_covered"] == ["built", "crops", "water"]

    manifest_path = output_dir / "arcgis_replacement_annotation_manifest.csv"
    rows = list(csv.DictReader(manifest_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 3
    assert {row["dominant_esri_class"] for row in rows} == {
        "water",
        "crops",
        "built",
    }
    for row in rows:
        assert row["manual_mask_path"] == ""
        assert row["paper12_mask_path"] == ""
        assert row["review_status"] == "pending_manual_annotation"
        assert (output_dir / row["arcgis_mask_path"]).exists()
        assert (output_dir / row["source_rgb_path"]).exists()
        assert (output_dir / row["preview_path"]).exists()

    summary_payload = json.loads(
        (output_dir / "packet_summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload == summary
    assert (output_dir / "annotation_readme.md").exists()


def test_packet_builder_cli_writes_summary_json(tmp_path):
    index_path = _write_index(tmp_path)
    output_dir = tmp_path / "packet_cli"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/prepare_arcgis_replacement_validation_packet.py"),
            "--index",
            str(index_path),
            "--output-dir",
            str(output_dir),
            "--sample-count",
            "2",
            "--year",
            "2022",
            "--required-classes",
            "water,built",
            "--seed",
            "3",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "packet_summary.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == 2
    assert summary["critical_classes_covered"] == ["built", "water"]
    assert summary["evaluator_ready"] is False


def test_validation_protocol_points_to_packet_builder():
    protocol_path = REPO_ROOT / "paper12_results/linhe_manual_validation_protocol.json"
    supplementary_path = (
        REPO_ROOT
        / "submission/paper12_isprs_jprs_20260606/06_supplementary_material"
        / "paper12_results/linhe_manual_validation_protocol.json"
    )
    template_path = REPO_ROOT / "paper12_results/arcgis_replacement_validation_template.json"

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    supplementary = json.loads(supplementary_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))

    assert protocol == supplementary
    assert (
        protocol["packet_builder_script"]
        == "scripts/prepare_arcgis_replacement_validation_packet.py"
    )
    assert any(
        "scripts/prepare_arcgis_replacement_validation_packet.py" in action
        for action in protocol["next_actions"]
    )
    assert any(
        "scripts/prepare_arcgis_replacement_validation_packet.py" in action
        for action in template["next_actions"]
    )
