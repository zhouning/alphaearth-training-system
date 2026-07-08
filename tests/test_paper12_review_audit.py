from __future__ import annotations

import json
from pathlib import Path

import pytest

from geoadapter.bench.paper12_audit import build_review_audit, write_review_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_review_audit_computes_capacity_and_channel_bridge_boundaries():
    audit = build_review_audit(REPO_ROOT)

    capacity = audit["capacity_audit"]
    assert capacity["largest_lora_method"] == "lora_split_qkv_r64"
    assert capacity["smallest_houlsby_method"] == "houlsby_d8"
    assert capacity["largest_lora_params"] == 4_726_282
    assert capacity["smallest_houlsby_params"] == 164_458
    assert capacity["largest_lora_oa"] == pytest.approx(0.7078395061728395)
    assert capacity["smallest_houlsby_oa"] == pytest.approx(0.8644444444444445)
    assert capacity["largest_lora_minus_smallest_houlsby_oa"] == pytest.approx(
        -0.1566049382716049
    )
    assert capacity["lora_params_over_houlsby_params"] == pytest.approx(
        28.73853506670396
    )
    assert capacity["lora_oa_monotonic_non_decreasing"] is False
    assert capacity["houlsby_oa_monotonic_non_decreasing"] is True

    bridge = audit["channel_bridge_audit"]
    assert bridge["learned_minus_zero_pad_linear_probe_oa"] == pytest.approx(
        0.07339506172839518
    )
    assert bridge["learned_minus_zero_pad_houlsby_oa"] == pytest.approx(
        0.010864197530864213
    )
    assert bridge["zero_pad_houlsby_minus_linear_probe_oa"] == pytest.approx(
        0.20938271604938277
    )
    assert bridge["learned_houlsby_minus_linear_probe_oa"] == pytest.approx(
        0.1468518518518518
    )
    assert bridge["houlsby_over_linear_ordering_preserved"] is True


def test_review_audit_records_loveda_full_finetune_boundary_and_diagnostics():
    audit = build_review_audit(REPO_ROOT)

    loveda = audit["loveda_audit"]
    assert loveda["u2r"]["houlsby_minus_full_finetune_miou"] == pytest.approx(
        0.02388883670235656
    )
    assert loveda["r2u"]["houlsby_minus_full_finetune_miou"] == pytest.approx(
        0.07280095923900659
    )
    assert loveda["u2r"]["full_finetune_above_small_peft_cluster"] is True
    assert loveda["r2u"]["full_finetune_above_small_peft_cluster"] is True
    assert loveda["u2r"]["full_finetune_below_houlsby"] is True
    assert loveda["r2u"]["full_finetune_below_houlsby"] is True

    diagnostic = audit["loveda_diagnostic"]
    assert diagnostic["best_lora_diag_method"] == "lora_r8_diag"
    assert diagnostic["best_lora_diag_miou"] == pytest.approx(0.0901)
    assert diagnostic["nonzero_iou_classes"] == 5
    assert diagnostic["classes_above_0_05_iou"] == 2
    assert diagnostic["best_lora_diag_dominant_prediction_share"] == pytest.approx(
        0.8782
    )


def test_review_audit_records_model_scope_label_and_decoder_boundaries():
    audit = build_review_audit(REPO_ROOT)

    scope = audit["model_scope_audit"]
    assert scope["backbones_evaluated"] == ["Prithvi-100M", "satmae_vit_base"]
    assert scope["backbone_count"] == 2
    assert scope["second_backbone_results_completed"] is True
    assert scope["general_geo_fm_ranking_supported"] is False
    assert scope["bigearthnet_subset"] == "10K train / 5K validation"

    second = audit["second_backbone_audit"]
    assert second["schema"] == "paper12.second_backbone_eurosat_summary.v1"
    assert second["row_count"] == 18
    assert second["best_methods_by_modality"] == {
        "rgb": "satmae_houlsby_d64",
        "s2_full": "satmae_houlsby_d64",
    }
    assert second["houlsby_s2_full_oa"] == pytest.approx(0.9066666666666667)
    assert second["houlsby_rgb_oa"] == pytest.approx(0.8393827160493827)
    assert second["s2_full_houlsby_minus_lora_oa"] == pytest.approx(0.3655555555555555)
    assert second["rgb_houlsby_minus_lora_oa"] == pytest.approx(0.497037037037037)

    linhe = audit["linhe_label_audit"]
    assert linhe["supervisory_label_source"] == "Esri 2022 LULC"
    assert linhe["independent_manual_ground_truth"] is False
    assert linhe["synthetic_control_is_manual_validation"] is False
    assert linhe["houlsby_minus_linear_miou"] == pytest.approx(0.115684)
    assert linhe["synthetic_weak_label_delta_miou"] == pytest.approx(0.017)
    assert linhe["esri_delta_over_synthetic_delta"] == pytest.approx(
        6.804941176470588
    )
    assert linhe["lora_mean_within_0_001_of_linear"] is True

    decoder = audit["segmentation_decoder_audit"]
    assert decoder["decoder_type"] == "single_1x1_conv_plus_bilinear_upsample"
    assert decoder["patch_size"] == 16
    assert decoder["landcoverai_head_params"] == 4_614
    assert decoder["linhe_head_params"] == 4_614
    assert decoder["loveda_head_params"] == 5_383
    assert decoder["linear_probe_params_match_head_only"] is True
    assert decoder["absolute_miou_can_be_decoder_limited"] is True


def test_review_audit_records_arcgis_replacement_boundary():
    audit = build_review_audit(REPO_ROOT)

    replacement = audit["arcgis_replacement_audit"]
    assert replacement["schema"] == "paper12.arcgis_replacement_validation.v1"
    assert replacement["decision_status"] == "not_validated"
    assert replacement["evidence_level"] == "weak_supervision_evidence"
    assert replacement["replacement_claim_supported"] is False
    assert replacement["arcgis_replacement_ready"] is False
    assert replacement["manual_ground_truth_available"] is False
    assert replacement["arcgis_reference_available"] is False
    assert replacement["paper12_model_checkpoint_available"] is False
    assert replacement["same_area_same_time_same_taxonomy"] is False
    assert replacement["paired_model_outputs_available"] is False
    assert replacement["current_boundary"] == (
        "Paper12 supports local weak-supervision adaptation evidence, not a "
        "validated ArcGIS replacement claim."
    )
    assert "independent_manual_ground_truth" in replacement["missing_evidence"]
    assert "arcgis_reference_output" in replacement["missing_evidence"]

def test_write_review_audit_creates_deterministic_json(tmp_path):
    output = tmp_path / "review_audit_summary.json"
    audit = write_review_audit(REPO_ROOT, output)

    assert output.exists()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == audit
    assert loaded["audit_schema_version"] == 2
    assert loaded["source_files"] == [
        "paper12_results/peft_capacity_sweep_summary.json",
        "paper12_results/eurosat_channel_bridge_summary.json",
        "paper12_results/second_backbone_eurosat.json",
        "paper12_results/second_backbone_eurosat_summary.json",
        "results/loveda/loveda_lulc_seg.json",
        "paper12_results/loveda_full_finetune_summary.json",
        "results/loveda/loveda_u2r_diag.json",
        "paper12_results/landcoverai_segmentation.json",
        "linhe_results/linhe_lulc_seg.json",
        "paper12_results/arcgis_replacement_validation_template.json",
    ]


