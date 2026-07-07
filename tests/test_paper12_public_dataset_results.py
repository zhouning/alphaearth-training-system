from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import pytest
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
ACTION_REQUIRED = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "00_ACTION_REQUIRED.md"
)
SUBMISSION_METHOD_SECTION = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "02_latex_source"
    / "sections"
    / "method.tex"
)
MANUSCRIPT_LOVEDA_SECTION = REPO_ROOT / "paper12" / "sections" / "linhe_validation.tex"
SUBMISSION_LOVEDA_SECTION = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "02_latex_source"
    / "sections"
    / "linhe_validation.tex"
)
PAPER12_WORD_EXPORT = REPO_ROOT / "paper12" / "paper12_english_for_word.tex"
PAPER12_COVER_LETTER = REPO_ROOT / "paper12" / "cover_letter.tex"
PAPER12_RSE_COVER_LETTER = REPO_ROOT / "paper12" / "cover_letter_rse.tex"
SUBMISSION_COVER_LETTER_MD = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "03_cover_letter"
    / "cover_letter_isprs_jprs.md"
)
SUBMISSION_COVER_LETTER_TEX = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "03_cover_letter"
    / "cover_letter_isprs_jprs.tex"
)
SUBMISSION_ABSTRACT_PLAIN_TEXT = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "05_highlights_abstract_keywords"
    / "abstract_plain_text.md"
)
SUBMISSION_HIGHLIGHTS = (
    REPO_ROOT
    / "submission"
    / "paper12_isprs_jprs_20260606"
    / "05_highlights_abstract_keywords"
    / "highlights.md"
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


def test_loveda_full_finetune_summary_records_completed_two_direction_results():
    u2r_rows = json.loads(
        (PAPER12_RESULTS / "loveda_full_finetune_u2r.json").read_text(encoding="utf-8")
    )
    r2u_rows = json.loads(
        (PAPER12_RESULTS / "loveda_full_finetune_r2u.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (PAPER12_RESULTS / "loveda_full_finetune_summary.json").read_text(
            encoding="utf-8"
        )
    )

    for direction, rows in {"u2r": u2r_rows, "r2u": r2u_rows}.items():
        assert len(rows) == 3
        assert [row["seed"] for row in rows] == [42, 123, 456]
        miou = [float(row["mIoU"]) for row in rows]
        assert summary[direction]["status"] == "completed"
        assert summary[direction]["mIoU_mean"] == mean(miou)
        assert summary[direction]["mIoU_std"] == stdev(miou)
        assert summary[direction]["mIoU_values"] == miou
        assert summary[direction]["seeds"] == [42, 123, 456]


def test_required_experiments_tracks_colab_result_status():
    text = REQUIRED_EXPERIMENTS.read_text(encoding="utf-8")

    assert "EuroSAT channel-bridge ablation: completed and manuscript-ready." in text
    assert "LoveDA full fine-tuning U->R: completed" in text
    assert "LoveDA full fine-tuning R->U: completed" in text
    assert "The EuroSAT channel-bridge rerun has completed and produced a 12-row JSON plus summary." in text
    assert "PEFT capacity sweep: completed and manuscript-ready." in text
    assert "colab/paper12_peft_capacity_sweep_colab.ipynb" in text
    assert "peft_capacity_sweep_summary.json" in text
    assert "The EuroSAT PEFT capacity sweep has completed and produced a 30-row JSON plus summary." in text


def test_capacity_sweep_is_completed_and_marked_manuscript_ready():
    required = REQUIRED_EXPERIMENTS.read_text(encoding="utf-8")
    action = ACTION_REQUIRED.read_text(encoding="utf-8")
    method = SUBMISSION_METHOD_SECTION.read_text(encoding="utf-8")

    assert "Completed EuroSAT Capacity-Audit Extension" in method
    assert "30-run EuroSAT capacity sweep" in method
    assert "PEFT capacity sweep: completed and manuscript-ready." in required
    assert "Prepared and awaiting Colab execution." not in required
    assert "PEFT capacity-sweep notebook and config are prepared" not in required
    assert "do not cite the capacity curve as completed evidence" not in action
    assert "peft_capacity_sweep.json" in action
    assert "peft_capacity_sweep_summary.json" in action
    assert "review\\_audit\\_summary.json" in method


def test_peft_capacity_sweep_summary_matches_raw_results():
    rows = json.loads(
        (PAPER12_RESULTS / "peft_capacity_sweep.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (PAPER12_RESULTS / "peft_capacity_sweep_summary.json").read_text(
            encoding="utf-8"
        )
    )

    expected_methods = {
        "linear_probe",
        "lora_split_qkv_r4",
        "lora_split_qkv_r8",
        "lora_split_qkv_r16",
        "lora_split_qkv_r32",
        "lora_split_qkv_r64",
        "houlsby_d8",
        "houlsby_d16",
        "houlsby_d32",
        "houlsby_d64",
    }

    assert len(rows) == 30
    assert set(summary) == expected_methods

    for method in expected_methods:
        method_rows = [row for row in rows if row["method"] == method]
        assert [row["seed"] for row in method_rows] == [42, 123, 456]
        assert summary[method]["seeds"] == [42, 123, 456]
        params = {int(row["trainable_params"]) for row in method_rows}
        assert len(params) == 1
        assert summary[method]["trainable_params"] == params.pop()
        oa = [float(row["overall_accuracy"]) for row in method_rows]
        macro_f1 = [float(row["macro_f1"]) for row in method_rows]
        assert summary[method]["overall_accuracy_mean"] == pytest.approx(mean(oa))
        assert summary[method]["overall_accuracy_std"] == pytest.approx(stdev(oa))
        assert summary[method]["macro_f1_mean"] == pytest.approx(mean(macro_f1))
        assert summary[method]["macro_f1_std"] == pytest.approx(stdev(macro_f1))

def test_loveda_table_includes_completed_full_finetuning_baseline():
    expected_row = (
        "Full fine-tuning & 86{,}242{,}567 & 0.1145 $\\pm$ 0.0028 & "
        "$+0.0287$ & 0.1391 $\\pm$ 0.0085"
    )
    expected_run_count = (
        "giving 30 PEFT runs in total. We additionally run full fine-tuning under "
        "the same three seeds in both directions, adding six unfrozen-baseline runs."
    )

    for path in [MANUSCRIPT_LOVEDA_SECTION, SUBMISSION_LOVEDA_SECTION]:
        text = path.read_text(encoding="utf-8")
        assert expected_row in text
        assert expected_run_count in text


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
        "peft_capacity_sweep.json",
        "peft_capacity_sweep_summary.json",
        "loveda_full_finetune_r2u.json",
        "loveda_full_finetune_u2r.json",
        "loveda_full_finetune_summary.json",
        "review_audit_summary.json",
    ]:
        assert (SUPPLEMENTARY_RESULTS / name).read_text(encoding="utf-8") == (
            PAPER12_RESULTS / name
        ).read_text(encoding="utf-8")


def test_eurosat_channel_bridge_rerun_records_final_counts():
    rows = json.loads(
        (PAPER12_RESULTS / "eurosat_channel_bridge.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (PAPER12_RESULTS / "eurosat_channel_bridge_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(rows) == 12
    assert summary["learned_bridge_houlsby"]["seeds"] == [42, 123, 456]
    assert summary["zero_pad_linear_probe"]["seeds"] == [42, 123, 456]

def test_manuscript_bounds_reviewer_sensitive_claims_after_audit_extension():
    manuscript_paths = [
        REPO_ROOT / "paper12" / "sections" / "introduction.tex",
        REPO_ROOT / "paper12" / "sections" / "method.tex",
        REPO_ROOT / "paper12" / "sections" / "results.tex",
        REPO_ROOT / "paper12" / "sections" / "segmentation.tex",
        REPO_ROOT / "paper12" / "sections" / "linhe_validation.tex",
        REPO_ROOT / "paper12" / "sections" / "discussion.tex",
        REPO_ROOT / "paper12" / "sections" / "appendix.tex",
        REPO_ROOT / "submission" / "paper12_isprs_jprs_20260606" / "02_latex_source" / "sections" / "introduction.tex",
        REPO_ROOT / "submission" / "paper12_isprs_jprs_20260606" / "02_latex_source" / "sections" / "method.tex",
        REPO_ROOT / "submission" / "paper12_isprs_jprs_20260606" / "02_latex_source" / "sections" / "results.tex",
        REPO_ROOT / "submission" / "paper12_isprs_jprs_20260606" / "02_latex_source" / "sections" / "segmentation.tex",
        REPO_ROOT / "submission" / "paper12_isprs_jprs_20260606" / "02_latex_source" / "sections" / "linhe_validation.tex",
        REPO_ROOT / "submission" / "paper12_isprs_jprs_20260606" / "02_latex_source" / "sections" / "discussion.tex",
        REPO_ROOT / "submission" / "paper12_isprs_jprs_20260606" / "02_latex_source" / "sections" / "appendix.tex",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in manuscript_paths)

    forbidden_phrases = [
        "paired label-quality control study",
        "LoRA fails on Prithvi-100M",
        "LoRA collapses to linear probing on production data, exactly as on public benchmarks",
        "structural limitation of PyTorch-implemented LoRA on Prithvi-100M",
        "strongest evidence to date",
        "close that gap",
        "directly close the ``segmentation remains open'' caveat",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in combined

    required_phrases = [
        "synthetic weak-label control",
        "not an independent manual validation set",
        "single-backbone Prithvi-100M setting",
        "lightweight linear segmentation decoder may limit absolute mIoU",
        "review\\_audit\\_summary.json",
        "model-scope, label-source, and decoder-capacity checks",
    ]
    for phrase in required_phrases:
        assert phrase in combined


def test_submission_side_materials_bound_reviewer_sensitive_claims():
    side_material_paths = [
        PAPER12_WORD_EXPORT,
        PAPER12_COVER_LETTER,
        PAPER12_RSE_COVER_LETTER,
        SUBMISSION_COVER_LETTER_MD,
        SUBMISSION_COVER_LETTER_TEX,
        SUBMISSION_ABSTRACT_PLAIN_TEXT,
        SUBMISSION_HIGHLIGHTS,
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in side_material_paths)

    forbidden_phrases = [
        "LoRA fails on Prithvi-100M",
        "label-quality control",
        "adapter-capacity limit",
        "strongest evidence",
        "segmentation remains open",
        "only two benchmark datasets",
        "first systematic benchmark",
        "totaling 100 experiments",
        "across all four datasets",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in combined

    required_phrases = [
        "synthetic weak-label control",
        "not an independent manual validation",
        "single-backbone Prithvi-100M",
        "adapter-capacity hypothesis",
        "LoveDA",
    ]
    for phrase in required_phrases:
        assert phrase in combined

def test_paper12_cover_letter_templates_avoid_optional_linebreak_placeholders():
    cover_letter_paths = [
        PAPER12_COVER_LETTER,
        PAPER12_RSE_COVER_LETTER,
        SUBMISSION_COVER_LETTER_TEX,
    ]

    for path in cover_letter_paths:
        text = path.read_text(encoding="utf-8")
        assert "\\ [" not in text

def test_paper12_cover_letters_avoid_unbreakable_code_tokens():
    cover_letter_paths = [
        PAPER12_COVER_LETTER,
        PAPER12_RSE_COVER_LETTER,
        SUBMISSION_COVER_LETTER_TEX,
    ]

    for path in cover_letter_paths:
        text = path.read_text(encoding="utf-8")
        assert r"\texttt{nn.MultiheadAttention}" not in text
