from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "geoadapter" / "bench" / "configs"
PAPER12_RESULTS = REPO_ROOT / "paper12_results"
SUPPLEMENTARY_RESULTS = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "06_supplementary_material"
    / "paper12_results"
)
REQUIRED_EXPERIMENTS = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "REQUIRED_EXPERIMENTS_ISPRS.md"
)


def test_eurosat_channel_bridge_summary_matches_raw_results():
    rows = json.loads(
        (PAPER12_RESULTS / "eurosat_channel_bridge.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (PAPER12_RESULTS / "eurosat_channel_bridge_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(rows) == 12
    expected_methods = {
        "zero_pad_linear_probe",
        "learned_bridge_linear_probe",
        "zero_pad_houlsby",
        "learned_bridge_houlsby",
    }
    assert set(summary) == expected_methods

    for method in expected_methods:
        method_rows = [row for row in rows if row["method"] == method]
        assert [row["seed"] for row in method_rows] == [42, 123, 456]
        assert summary[method]["seeds"] == [42, 123, 456]
        oa = [float(row["overall_accuracy"]) for row in method_rows]
        macro_f1 = [float(row["macro_f1"]) for row in method_rows]
        assert summary[method]["overall_accuracy_mean"] == mean(oa)
        assert summary[method]["overall_accuracy_std"] == stdev(oa)
        assert summary[method]["macro_f1_mean"] == mean(macro_f1)
        assert summary[method]["macro_f1_std"] == stdev(macro_f1)


def test_loveda_full_finetune_summary_records_completed_u2r_only():
    rows = json.loads(
        (PAPER12_RESULTS / "loveda_full_finetune_u2r.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (PAPER12_RESULTS / "loveda_full_finetune_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(rows) == 3
    assert [row["seed"] for row in rows] == [42, 123, 456]
    miou = [float(row["mIoU"]) for row in rows]
    assert summary["u2r"]["status"] == "completed"
    assert summary["u2r"]["mIoU_mean"] == mean(miou)
    assert summary["u2r"]["mIoU_std"] == stdev(miou)
    assert summary["u2r"]["seeds"] == [42, 123, 456]
    assert summary["r2u"]["status"] == "not_available_in_drive_at_update_time"


def test_required_experiments_tracks_colab_result_status():
    text = REQUIRED_EXPERIMENTS.read_text(encoding="utf-8")

    assert "EuroSAT channel-bridge ablation: completed only as pre-fix archive data" in text
    assert "LoveDA full fine-tuning U->R: completed" in text
    assert "LoveDA full fine-tuning R->U: still open" in text
    assert "rerun required for manuscript evidence" in text


def test_public_dataset_colab_configs_load_staged_prithvi_checkpoint():
    expected_checkpoint = "data/weights/prithvi/Prithvi_100M.pt"

    for name in [
        "eurosat_channel_bridge.yaml",
        "loveda_lulc_full_finetune_u2r.yaml",
        "loveda_lulc_full_finetune_r2u.yaml",
    ]:
        cfg = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert cfg["prithvi"]["pretrained"] is True
        assert cfg["prithvi"]["checkpoint"] == expected_checkpoint


def test_public_dataset_results_are_mirrored_in_supplementary_package():
    for name in [
        "eurosat_channel_bridge.json",
        "eurosat_channel_bridge_summary.json",
        "loveda_full_finetune_u2r.json",
        "loveda_full_finetune_summary.json",
    ]:
        assert (SUPPLEMENTARY_RESULTS / name).read_text(encoding="utf-8") == (
            PAPER12_RESULTS / name
        ).read_text(encoding="utf-8")
