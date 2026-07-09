from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_linhe_manual_validation_protocol_and_manifest_schema_exist():
    protocol_path = REPO_ROOT / "paper12_results/linhe_manual_validation_protocol.json"
    manifest_path = (
        REPO_ROOT / "paper12_results/linhe_manual_validation_manifest_template.csv"
    )
    supplementary_protocol_path = (
        REPO_ROOT
        / "submission/paper12_isprs_jprs_20260606/06_supplementary_material"
        / "paper12_results/linhe_manual_validation_protocol.json"
    )
    supplementary_manifest_path = (
        REPO_ROOT
        / "submission/paper12_isprs_jprs_20260606/06_supplementary_material"
        / "paper12_results/linhe_manual_validation_manifest_template.csv"
    )

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    supplementary_protocol = json.loads(
        supplementary_protocol_path.read_text(encoding="utf-8")
    )

    assert protocol == supplementary_protocol
    assert protocol["schema"] == "paper12.linhe_manual_validation_protocol.v1"
    assert protocol["decision_status_before_evidence"] == "not_validated"
    assert protocol["required_manifest_columns"] == [
        "sample_id",
        "manual_mask_path",
        "manual_label",
        "arcgis_mask_path",
        "arcgis_label",
        "paper12_mask_path",
        "paper12_label",
    ]
    assert protocol["required_evidence"] == [
        "independent_manual_ground_truth",
        "arcgis_reference_output",
        "paper12_checkpoint_backed_output",
        "same_area_same_time_same_taxonomy_pairing",
    ]
    assert protocol["default_critical_classes"] == ["water", "crops", "built"]
    assert protocol["recommended_statistical_controls"] == {
        "paired_delta_bootstrap_unit": "manifest_row",
        "paired_delta_bootstrap_iterations": 1000,
        "confidence_level": 0.95,
        "minimum_candidate_manifest_rows": 30,
    }
    assert protocol["coverage_diagnostics"] == {
        "critical_class_support": "valid manual pixels or scalar labels per critical class",
        "critical_class_row_support": "manifest rows containing each critical class after ignore-index filtering",
    }

    header = next(csv.reader(manifest_path.read_text(encoding="utf-8").splitlines()))
    supplementary_header = next(
        csv.reader(supplementary_manifest_path.read_text(encoding="utf-8").splitlines())
    )

    assert header == supplementary_header
    assert header[:7] == protocol["required_manifest_columns"]
    assert header[7:] == [
        "scene_id",
        "x",
        "y",
        "dominant_esri_class",
        "dominant_paper12_class",
        "annotator_id",
        "review_status",
    ]


def test_tiny_mask_metrics_compare_arcgis_and_paper12_against_manual():
    from scripts.evaluate_arcgis_replacement import compute_replacement_metrics

    manual = np.array([[0, 1, 2], [0, 1, 2]])
    arcgis = np.array([[0, 1, 2], [0, 0, 2]])
    paper12 = np.array([[0, 1, 1], [0, 1, 2]])

    result = compute_replacement_metrics(
        manual=manual,
        arcgis=arcgis,
        paper12=paper12,
        class_names=["water", "crops", "built"],
    )

    assert result["schema"] == "paper12.arcgis_replacement_evaluation.v1"
    assert result["pixel_count"] == 6

    arcgis_metrics = result["arcgis_vs_manual"]
    assert arcgis_metrics["confusion_matrix"] == [[2, 0, 0], [1, 1, 0], [0, 0, 2]]
    assert arcgis_metrics["overall_accuracy"] == pytest.approx(5 / 6)
    assert arcgis_metrics["macro_f1"] == pytest.approx((0.8 + (2 / 3) + 1.0) / 3)
    assert arcgis_metrics["per_class_iou"] == {
        "water": pytest.approx(2 / 3),
        "crops": pytest.approx(1 / 2),
        "built": pytest.approx(1.0),
    }
    assert arcgis_metrics["miou"] == pytest.approx((2 / 3 + 1 / 2 + 1.0) / 3)

    paper12_metrics = result["paper12_vs_manual"]
    assert paper12_metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 1, 1]]
    assert paper12_metrics["overall_accuracy"] == pytest.approx(5 / 6)
    assert paper12_metrics["macro_f1"] == pytest.approx((1.0 + 0.8 + (2 / 3)) / 3)
    assert paper12_metrics["per_class_iou"] == {
        "water": pytest.approx(1.0),
        "crops": pytest.approx(2 / 3),
        "built": pytest.approx(1 / 2),
    }
    assert paper12_metrics["miou"] == pytest.approx((1.0 + 2 / 3 + 1 / 2) / 3)

    assert result["paired_delta"] == {
        "overall_accuracy": pytest.approx(0.0),
        "macro_f1": pytest.approx(0.0),
        "miou": pytest.approx(0.0),
    }

def test_decision_statuses_are_conservative():
    from scripts.evaluate_arcgis_replacement import decide_replacement_status

    base_metrics = {
        "arcgis_vs_manual": {
            "miou": 0.5,
            "per_class_iou": {"water": 0.4, "crops": 0.5, "built": 0.6},
        },
        "paper12_vs_manual": {
            "miou": 0.5,
            "per_class_iou": {"water": 0.4, "crops": 0.55, "built": 0.6},
        },
    }

    candidate = decide_replacement_status(
        base_metrics,
        critical_classes=["water", "crops", "built"],
        tolerance=0.0,
    )

    assert candidate["decision_status"] == "replacement_candidate"
    assert candidate["replacement_claim_supported"] is True
    assert candidate["arcgis_replacement_ready"] is True
    assert candidate["reasons"] == []

    weaker_metrics = {
        "arcgis_vs_manual": {
            "miou": 0.5,
            "per_class_iou": {"water": 0.4, "crops": 0.5, "built": 0.6},
        },
        "paper12_vs_manual": {
            "miou": 0.49,
            "per_class_iou": {"water": 0.4, "crops": 0.49, "built": 0.6},
        },
    }

    partial = decide_replacement_status(
        weaker_metrics,
        critical_classes=["water", "crops", "built"],
        tolerance=0.0,
    )

    assert partial["decision_status"] == "partial"
    assert partial["replacement_claim_supported"] is False
    assert partial["arcgis_replacement_ready"] is False
    assert "paper12_below_arcgis_miou" in partial["reasons"]
    assert "paper12_below_arcgis_critical_class:crops" in partial["reasons"]

    not_validated = decide_replacement_status(None)
    assert not_validated["decision_status"] == "not_validated"
    assert not_validated["replacement_claim_supported"] is False
    assert not_validated["arcgis_replacement_ready"] is False
    assert "missing_paired_evidence" in not_validated["reasons"]


def test_manifest_missing_paper12_evidence_remains_not_validated(tmp_path):
    from scripts.evaluate_arcgis_replacement import evaluate_manifest

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,manual_mask_path,manual_label,arcgis_mask_path,arcgis_label,"
        "paper12_mask_path,paper12_label\n"
        "s1,,0,,0,,\n",
        encoding="utf-8",
    )

    result = evaluate_manifest(
        manifest,
        class_names=["water", "crops", "built"],
    )

    assert result["decision_status"] == "not_validated"
    assert result["replacement_claim_supported"] is False
    assert result["manifest_row_count"] == 1
    assert "paper12_output" in result["missing_evidence"]
    assert result["metrics"] is None


def test_manifest_bootstrap_reports_row_level_paired_delta_intervals(tmp_path):
    from scripts.evaluate_arcgis_replacement import evaluate_manifest

    samples = {
        "manual_a.npy": np.array([[0, 1], [1, 2]]),
        "arcgis_a.npy": np.array([[0, 1], [0, 2]]),
        "paper12_a.npy": np.array([[0, 1], [1, 2]]),
        "manual_b.npy": np.array([[0, 1], [2, 2]]),
        "arcgis_b.npy": np.array([[0, 1], [2, 2]]),
        "paper12_b.npy": np.array([[0, 2], [2, 2]]),
        "manual_c.npy": np.array([[0, 0], [1, 2]]),
        "arcgis_c.npy": np.array([[0, 2], [1, 2]]),
        "paper12_c.npy": np.array([[0, 0], [1, 2]]),
    }
    for name, mask in samples.items():
        np.save(tmp_path / name, mask)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,manual_mask_path,manual_label,arcgis_mask_path,arcgis_label,"
        "paper12_mask_path,paper12_label\n"
        "a,manual_a.npy,,arcgis_a.npy,,paper12_a.npy,\n"
        "b,manual_b.npy,,arcgis_b.npy,,paper12_b.npy,\n"
        "c,manual_c.npy,,arcgis_c.npy,,paper12_c.npy,\n",
        encoding="utf-8",
    )

    result = evaluate_manifest(
        manifest,
        class_names=["water", "crops", "built"],
        bootstrap_iterations=40,
        bootstrap_seed=7,
    )
    repeated = evaluate_manifest(
        manifest,
        class_names=["water", "crops", "built"],
        bootstrap_iterations=40,
        bootstrap_seed=7,
    )

    bootstrap = result["bootstrap"]
    assert bootstrap == repeated["bootstrap"]
    assert bootstrap["schema"] == "paper12.arcgis_replacement_bootstrap.v1"
    assert bootstrap["sample_unit"] == "manifest_row"
    assert bootstrap["iterations"] == 40
    assert bootstrap["seed"] == 7
    assert bootstrap["confidence_level"] == pytest.approx(0.95)
    assert bootstrap["row_count"] == 3

    ci = bootstrap["paired_delta_ci"]
    assert set(ci) == {"overall_accuracy", "macro_f1", "miou"}
    for metric_ci in ci.values():
        assert metric_ci["lower"] <= metric_ci["mean"] <= metric_ci["upper"]
        assert metric_ci["point_estimate"] == pytest.approx(
            result["metrics"]["paired_delta"][metric_ci["metric"]]
        )


def test_perfect_small_manifest_is_not_a_replacement_candidate_by_default(tmp_path):
    from scripts.evaluate_arcgis_replacement import evaluate_manifest

    manual = np.array([[0, 1, 2], [0, 1, 2]])
    np.save(tmp_path / "manual.npy", manual)
    np.save(tmp_path / "arcgis.npy", manual)
    np.save(tmp_path / "paper12.npy", manual)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,manual_mask_path,manual_label,arcgis_mask_path,arcgis_label,"
        "paper12_mask_path,paper12_label\n"
        "s1,manual.npy,,arcgis.npy,,paper12.npy,\n",
        encoding="utf-8",
    )

    result = evaluate_manifest(manifest, class_names=["water", "crops", "built"])

    assert result["decision_status"] == "insufficient_sample_size"
    assert result["replacement_claim_supported"] is False
    assert result["arcgis_replacement_ready"] is False
    assert result["min_candidate_rows"] == 30
    assert "insufficient_manifest_rows:1<30" in result["reasons"]

    smoke_result = evaluate_manifest(
        manifest,
        class_names=["water", "crops", "built"],
        min_candidate_rows=1,
    )
    assert smoke_result["decision_status"] == "replacement_candidate"
    assert smoke_result["replacement_claim_supported"] is True


def test_candidate_requires_manual_coverage_for_each_critical_class(tmp_path):
    from scripts.evaluate_arcgis_replacement import evaluate_manifest

    rows = [
        "sample_id,manual_mask_path,manual_label,arcgis_mask_path,arcgis_label,"
        "paper12_mask_path,paper12_label\n"
    ]
    for index in range(30):
        label = index % 2
        rows.append(f"s{index},,{label},,{label},,{label}\n")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("".join(rows), encoding="utf-8")

    result = evaluate_manifest(
        manifest,
        class_names=["water", "crops", "built"],
        min_candidate_rows=30,
    )

    assert result["decision_status"] == "insufficient_class_coverage"
    assert result["replacement_claim_supported"] is False
    assert result["arcgis_replacement_ready"] is False
    assert result["critical_class_support"] == {
        "water": 15,
        "crops": 15,
        "built": 0,
    }
    assert "missing_manual_critical_class:built" in result["reasons"]


def test_evaluator_reports_critical_class_row_support(tmp_path):
    from scripts.evaluate_arcgis_replacement import evaluate_manifest

    rows = [
        "sample_id,manual_mask_path,manual_label,arcgis_mask_path,arcgis_label,"
        "paper12_mask_path,paper12_label\n"
    ]
    for index in range(30):
        if index == 0:
            label = 1
        elif index == 1:
            label = 2
        else:
            label = 0
        rows.append(f"s{index},,{label},,{label},,{label}\n")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("".join(rows), encoding="utf-8")

    result = evaluate_manifest(
        manifest,
        class_names=["water", "crops", "built"],
        min_candidate_rows=30,
    )

    assert result["decision_status"] == "replacement_candidate"
    assert result["critical_class_support"] == {
        "water": 28,
        "crops": 1,
        "built": 1,
    }
    assert result["critical_class_row_support"] == {
        "water": 28,
        "crops": 1,
        "built": 1,
    }


def test_cli_evaluates_manifest_and_writes_json(tmp_path):
    import subprocess
    import sys

    manual = np.array([[0, 1, 2], [0, 1, 2]])
    arcgis = manual.copy()
    paper12 = manual.copy()

    np.save(tmp_path / "manual.npy", manual)
    np.save(tmp_path / "arcgis.npy", arcgis)
    np.save(tmp_path / "paper12.npy", paper12)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,manual_mask_path,manual_label,arcgis_mask_path,arcgis_label,"
        "paper12_mask_path,paper12_label\n"
        "s1,manual.npy,,arcgis.npy,,paper12.npy,\n",
        encoding="utf-8",
    )
    output = tmp_path / "arcgis_replacement_eval.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluate_arcgis_replacement.py"),
            "--manifest",
            str(manifest),
            "--class-names",
            "water,crops,built",
            "--output",
            str(output),
            "--bootstrap-iterations",
            "20",
            "--bootstrap-seed",
            "11",
            "--min-candidate-rows",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "paper12.arcgis_replacement_evaluation.v1"
    assert payload["manifest_row_count"] == 1
    assert payload["decision_status"] == "replacement_candidate"
    assert payload["replacement_claim_supported"] is True
    assert payload["metrics"]["pixel_count"] == 6
    assert payload["metrics"]["paired_delta"]["miou"] == pytest.approx(0.0)
    assert payload["min_candidate_rows"] == 1
    assert payload["bootstrap"]["iterations"] == 20
    assert payload["bootstrap"]["seed"] == 11


def test_arcgis_replacement_template_points_to_evaluator_script():
    template_path = REPO_ROOT / "paper12_results/arcgis_replacement_validation_template.json"
    supplementary_path = (
        REPO_ROOT
        / "submission/paper12_isprs_jprs_20260606/06_supplementary_material"
        / "paper12_results/arcgis_replacement_validation_template.json"
    )

    template = json.loads(template_path.read_text(encoding="utf-8"))
    supplementary = json.loads(supplementary_path.read_text(encoding="utf-8"))

    assert template == supplementary
    assert template["decision_status"] == "not_validated"
    assert template["replacement_claim_supported"] is False
    assert "insufficient_class_coverage" in template["decision_rule"]
    assert any(
        "scripts/evaluate_arcgis_replacement.py" in action
        for action in template["next_actions"]
    )
    assert any("--bootstrap-iterations" in action for action in template["next_actions"])
    assert template["coverage_diagnostics"] == {
        "critical_class_support": "valid manual pixels or scalar labels per critical class",
        "critical_class_row_support": "manifest rows containing each critical class after ignore-index filtering",
    }
    assert template["recommended_statistical_controls"] == {
        "paired_delta_bootstrap_unit": "manifest_row",
        "paired_delta_bootstrap_iterations": 1000,
        "confidence_level": 0.95,
        "minimum_candidate_manifest_rows": 30,
    }
