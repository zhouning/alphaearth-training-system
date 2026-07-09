"""Paper12 review-audit summary from existing result artifacts.

This module does not run training. It derives reviewer-facing checks from the
JSON artifacts already produced by the benchmark and Colab reruns.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


SOURCE_FILES = [
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


SYNTHETIC_WEAK_LABEL_CONTROL = {
    "label_source": "synth: mean(RGB) >= 140",
    "linear_probe_miou": 0.706,
    "houlsby_miou": 0.723,
}


def _read_json(repo_root: Path, rel_path: str) -> Any:
    return json.loads((repo_root / rel_path).read_text(encoding="utf-8"))


def _is_monotonic_non_decreasing(values: list[float]) -> bool:
    return all(next_value >= value for value, next_value in zip(values, values[1:]))


def _mean_by_method(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(float(row["mIoU"]))
    return {method: mean(values) for method, values in grouped.items()}


def _mean_by_method_and_direction(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row["direction"], {}).setdefault(row["method"], []).append(
            float(row["mIoU"])
        )
    return {
        direction: {method: mean(values) for method, values in methods.items()}
        for direction, methods in grouped.items()
    }


def _capacity_audit(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lora_order = [
        "lora_split_qkv_r4",
        "lora_split_qkv_r8",
        "lora_split_qkv_r16",
        "lora_split_qkv_r32",
        "lora_split_qkv_r64",
    ]
    houlsby_order = ["houlsby_d8", "houlsby_d16", "houlsby_d32", "houlsby_d64"]
    lora_oa = [float(summary[name]["overall_accuracy_mean"]) for name in lora_order]
    houlsby_oa = [
        float(summary[name]["overall_accuracy_mean"]) for name in houlsby_order
    ]

    largest_lora = "lora_split_qkv_r64"
    smallest_houlsby = "houlsby_d8"
    largest_lora_params = int(summary[largest_lora]["trainable_params"])
    smallest_houlsby_params = int(summary[smallest_houlsby]["trainable_params"])
    largest_lora_oa = float(summary[largest_lora]["overall_accuracy_mean"])
    smallest_houlsby_oa = float(summary[smallest_houlsby]["overall_accuracy_mean"])

    return {
        "largest_lora_method": largest_lora,
        "smallest_houlsby_method": smallest_houlsby,
        "largest_lora_params": largest_lora_params,
        "smallest_houlsby_params": smallest_houlsby_params,
        "largest_lora_oa": largest_lora_oa,
        "smallest_houlsby_oa": smallest_houlsby_oa,
        "largest_lora_minus_smallest_houlsby_oa": largest_lora_oa
        - smallest_houlsby_oa,
        "lora_params_over_houlsby_params": largest_lora_params
        / smallest_houlsby_params,
        "lora_oa_monotonic_non_decreasing": _is_monotonic_non_decreasing(lora_oa),
        "houlsby_oa_monotonic_non_decreasing": _is_monotonic_non_decreasing(
            houlsby_oa
        ),
    }


def _channel_bridge_audit(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    zero_linear = float(summary["zero_pad_linear_probe"]["overall_accuracy_mean"])
    learned_linear = float(summary["learned_bridge_linear_probe"]["overall_accuracy_mean"])
    zero_houlsby = float(summary["zero_pad_houlsby"]["overall_accuracy_mean"])
    learned_houlsby = float(summary["learned_bridge_houlsby"]["overall_accuracy_mean"])
    return {
        "learned_minus_zero_pad_linear_probe_oa": learned_linear - zero_linear,
        "learned_minus_zero_pad_houlsby_oa": learned_houlsby - zero_houlsby,
        "zero_pad_houlsby_minus_linear_probe_oa": zero_houlsby - zero_linear,
        "learned_houlsby_minus_linear_probe_oa": learned_houlsby - learned_linear,
        "houlsby_over_linear_ordering_preserved": (
            zero_houlsby > zero_linear and learned_houlsby > learned_linear
        ),
    }


def _second_backbone_audit(
    raw_rows: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    groups = summary["groups"]
    by_key = {(item["method"], item["modality"]): item for item in groups}
    best_methods_by_modality = {
        item["modality"]: item["method"]
        for item in groups
        if int(item["rank_by_overall_accuracy"]) == 1
    }

    s2_houlsby = float(
        by_key[("satmae_houlsby_d64", "s2_full")]["overall_accuracy_mean"]
    )
    s2_lora = float(
        by_key[("satmae_lora_split_qkv_r8", "s2_full")]["overall_accuracy_mean"]
    )
    rgb_houlsby = float(
        by_key[("satmae_houlsby_d64", "rgb")]["overall_accuracy_mean"]
    )
    rgb_lora = float(
        by_key[("satmae_lora_split_qkv_r8", "rgb")]["overall_accuracy_mean"]
    )

    return {
        "schema": summary["schema"],
        "row_count": int(summary["row_count"]),
        "raw_row_count": len(raw_rows),
        "best_methods_by_modality": dict(sorted(best_methods_by_modality.items())),
        "houlsby_s2_full_oa": s2_houlsby,
        "houlsby_s2_full_macro_f1": float(
            by_key[("satmae_houlsby_d64", "s2_full")]["macro_f1_mean"]
        ),
        "houlsby_rgb_oa": rgb_houlsby,
        "houlsby_rgb_macro_f1": float(
            by_key[("satmae_houlsby_d64", "rgb")]["macro_f1_mean"]
        ),
        "s2_full_houlsby_minus_lora_oa": s2_houlsby - s2_lora,
        "rgb_houlsby_minus_lora_oa": rgb_houlsby - rgb_lora,
        "supports_second_backbone_consistency": (
            best_methods_by_modality
            == {"rgb": "satmae_houlsby_d64", "s2_full": "satmae_houlsby_d64"}
        ),
    }


def _loveda_audit(
    peft_rows: list[dict[str, Any]], full_summary: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    means = _mean_by_method_and_direction(peft_rows)
    result: dict[str, dict[str, Any]] = {}
    key_map = {"U->R": "u2r", "R->U": "r2u"}
    small_methods = ["linear_probe", "bitfit", "lora_r8", "geoadapter"]
    for direction, summary_key in key_map.items():
        direction_means = means[direction]
        full_miou = float(full_summary[summary_key]["mIoU_mean"])
        houlsby_miou = float(direction_means["houlsby"])
        small_cluster_max = max(float(direction_means[name]) for name in small_methods)
        result[summary_key] = {
            "houlsby_miou": houlsby_miou,
            "full_finetune_miou": full_miou,
            "small_peft_cluster_max_miou": small_cluster_max,
            "houlsby_minus_full_finetune_miou": houlsby_miou - full_miou,
            "full_finetune_minus_small_peft_cluster_miou": full_miou
            - small_cluster_max,
            "full_finetune_above_small_peft_cluster": full_miou > small_cluster_max,
            "full_finetune_below_houlsby": full_miou < houlsby_miou,
        }
    return result


def _loveda_diagnostic(diag_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(diag_rows, key=lambda row: float(row["mIoU"]))
    per_class = [float(value) for value in best["per_class_iou"]]
    return {
        "best_lora_diag_method": best["method"],
        "best_lora_diag_miou": float(best["mIoU"]),
        "nonzero_iou_classes": sum(value > 0.0 for value in per_class),
        "classes_above_0_05_iou": sum(value > 0.05 for value in per_class),
        "best_lora_diag_dominant_prediction_share": float(best["dom_pred_share"]),
    }


def _model_scope_audit(second_backbone_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "backbones_evaluated": ["Prithvi-100M", "satmae_vit_base"],
        "backbone_count": 2,
        "second_backbone_results_completed": (
            int(second_backbone_summary["row_count"]) == 18
        ),
        "general_geo_fm_ranking_supported": False,
        "bigearthnet_subset": "10K train / 5K validation",
        "ranking_scope": (
            "Prithvi-100M across the reported tasks, plus SatMAE-compatible "
            "EuroSAT validation for the PEFT ordering boundary"
        ),
    }



def _arcgis_replacement_audit(template: dict[str, Any]) -> dict[str, Any]:
    required = template["required_evidence"]
    return {
        "schema": template["schema"],
        "decision_status": template["decision_status"],
        "evidence_level": template["evidence_level"],
        "replacement_claim_supported": bool(template["replacement_claim_supported"]),
        "arcgis_replacement_ready": template["decision_status"] == "replacement_candidate",
        "manual_ground_truth_available": bool(required["manual_ground_truth_available"]),
        "arcgis_reference_available": bool(required["arcgis_reference_available"]),
        "paper12_model_checkpoint_available": bool(required["paper12_model_checkpoint_available"]),
        "same_area_same_time_same_taxonomy": bool(required["same_area_same_time_same_taxonomy"]),
        "paired_model_outputs_available": bool(required["paired_model_outputs_available"]),
        "missing_evidence": list(template["missing_evidence"]),
        "current_boundary": template["current_boundary"],
        "recommended_statistical_controls": dict(
            template.get("recommended_statistical_controls", {})
        ),
        "next_actions": list(template["next_actions"]),
    }


def _linhe_label_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means = _mean_by_method(rows)
    linear_miou = means["linear_probe"]
    houlsby_miou = means["houlsby"]
    lora_miou = means["lora_r8"]
    synthetic_delta = (
        SYNTHETIC_WEAK_LABEL_CONTROL["houlsby_miou"]
        - SYNTHETIC_WEAK_LABEL_CONTROL["linear_probe_miou"]
    )
    esri_delta = houlsby_miou - linear_miou

    return {
        "supervisory_label_source": "Esri 2022 LULC",
        "independent_manual_ground_truth": False,
        "synthetic_control_label_source": SYNTHETIC_WEAK_LABEL_CONTROL["label_source"],
        "synthetic_control_is_manual_validation": False,
        "linear_probe_miou": linear_miou,
        "houlsby_miou": houlsby_miou,
        "lora_miou": lora_miou,
        "houlsby_minus_linear_miou": esri_delta,
        "lora_minus_linear_miou": lora_miou - linear_miou,
        "lora_mean_within_0_001_of_linear": abs(lora_miou - linear_miou) <= 0.001,
        "synthetic_weak_label_delta_miou": synthetic_delta,
        "esri_delta_over_synthetic_delta": esri_delta / synthetic_delta,
    }


def _head_params(in_dim: int, num_classes: int) -> int:
    return in_dim * num_classes + num_classes


def _linear_probe_params(rows: list[dict[str, Any]], *, method: str = "linear_probe") -> int:
    params = {int(row["trainable_params"]) for row in rows if row["method"] == method}
    if len(params) != 1:
        raise ValueError(f"Expected one trainable-parameter count for {method}: {params}")
    return params.pop()


def _segmentation_decoder_audit(
    landcover_rows: list[dict[str, Any]],
    linhe_rows: list[dict[str, Any]],
    loveda_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    in_dim = 768
    patch_size = 16
    landcover_head_params = _head_params(in_dim, 6)
    linhe_head_params = _head_params(in_dim, 6)
    loveda_head_params = _head_params(in_dim, 7)
    observed = [
        _linear_probe_params(landcover_rows),
        _linear_probe_params(linhe_rows),
        _linear_probe_params(loveda_rows),
    ]
    expected = [landcover_head_params, linhe_head_params, loveda_head_params]

    return {
        "decoder_type": "single_1x1_conv_plus_bilinear_upsample",
        "decoder_is_lightweight_linear": True,
        "in_dim": in_dim,
        "patch_size": patch_size,
        "landcoverai_num_classes": 6,
        "linhe_num_classes": 6,
        "loveda_num_classes": 7,
        "landcoverai_head_params": landcover_head_params,
        "linhe_head_params": linhe_head_params,
        "loveda_head_params": loveda_head_params,
        "linear_probe_params_match_head_only": observed == expected,
        "absolute_miou_can_be_decoder_limited": True,
    }


def build_review_audit(repo_root: str | Path) -> dict[str, Any]:
    """Build the Paper12 review-audit summary from repository artifacts."""
    repo_root = Path(repo_root)
    capacity_summary = _read_json(repo_root, SOURCE_FILES[0])
    channel_summary = _read_json(repo_root, SOURCE_FILES[1])
    second_backbone_raw = _read_json(repo_root, SOURCE_FILES[2])
    second_backbone_summary = _read_json(repo_root, SOURCE_FILES[3])
    loveda_peft = _read_json(repo_root, SOURCE_FILES[4])
    loveda_full = _read_json(repo_root, SOURCE_FILES[5])
    loveda_diag = _read_json(repo_root, SOURCE_FILES[6])
    landcover_rows = _read_json(repo_root, SOURCE_FILES[7])
    linhe_rows = _read_json(repo_root, SOURCE_FILES[8])
    arcgis_replacement_template = _read_json(repo_root, SOURCE_FILES[9])

    return {
        "audit_schema_version": 2,
        "source_files": SOURCE_FILES,
        "capacity_audit": _capacity_audit(capacity_summary),
        "channel_bridge_audit": _channel_bridge_audit(channel_summary),
        "loveda_audit": _loveda_audit(loveda_peft, loveda_full),
        "loveda_diagnostic": _loveda_diagnostic(loveda_diag),
        "model_scope_audit": _model_scope_audit(second_backbone_summary),
        "second_backbone_audit": _second_backbone_audit(
            second_backbone_raw, second_backbone_summary
        ),
        "linhe_label_audit": _linhe_label_audit(linhe_rows),
        "arcgis_replacement_audit": _arcgis_replacement_audit(
            arcgis_replacement_template
        ),
        "segmentation_decoder_audit": _segmentation_decoder_audit(
            landcover_rows, linhe_rows, loveda_peft
        ),
    }


def write_review_audit(repo_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Write a deterministic JSON audit summary and return the same object."""
    audit = build_review_audit(repo_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output",
        default="paper12_results/review_audit_summary.json",
        help="Destination JSON path for the derived audit summary.",
    )
    args = parser.parse_args()
    write_review_audit(args.repo_root, args.output)


if __name__ == "__main__":
    main()