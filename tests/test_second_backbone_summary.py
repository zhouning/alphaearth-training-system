from __future__ import annotations

import json
from pathlib import Path

import pytest


def _rows():
    rows = []
    for modality in ("s2_full", "rgb"):
        for seed in (42, 123, 456):
            rows.append(
                {
                    "backbone": "satmae_vit_base",
                    "method": "satmae_linear_probe",
                    "modality": modality,
                    "seed": seed,
                    "trainable_params": 7690,
                    "overall_accuracy": 0.70 if modality == "s2_full" else 0.60,
                    "macro_f1": 0.69 if modality == "s2_full" else 0.59,
                }
            )
            rows.append(
                {
                    "backbone": "satmae_vit_base",
                    "method": "satmae_lora_split_qkv_r8",
                    "modality": modality,
                    "seed": seed,
                    "trainable_params": 155146,
                    "overall_accuracy": 0.72 if modality == "s2_full" else 0.62,
                    "macro_f1": 0.71 if modality == "s2_full" else 0.61,
                }
            )
            rows.append(
                {
                    "backbone": "satmae_vit_base",
                    "method": "satmae_houlsby_d64",
                    "modality": modality,
                    "seed": seed,
                    "trainable_params": 1197322,
                    "overall_accuracy": 0.80 if modality == "s2_full" else 0.68,
                    "macro_f1": 0.79 if modality == "s2_full" else 0.67,
                }
            )
    return rows


def test_build_second_backbone_summary_groups_and_ranks_methods():
    from geoadapter.bench.second_backbone_summary import build_second_backbone_summary

    summary = build_second_backbone_summary(_rows())

    assert summary["schema"] == "paper12.second_backbone_eurosat_summary.v1"
    assert summary["row_count"] == 18
    assert len(summary["groups"]) == 6

    by_key = {
        (item["method"], item["modality"]): item
        for item in summary["groups"]
    }
    houlsby_s2 = by_key[("satmae_houlsby_d64", "s2_full")]
    assert houlsby_s2["overall_accuracy_mean"] == pytest.approx(0.80)
    assert houlsby_s2["macro_f1_mean"] == pytest.approx(0.79)
    assert houlsby_s2["rank_by_overall_accuracy"] == 1
    assert houlsby_s2["seeds"] == [42, 123, 456]

    linear_rgb = by_key[("satmae_linear_probe", "rgb")]
    assert linear_rgb["rank_by_overall_accuracy"] == 3


def test_build_second_backbone_summary_requires_fields():
    from geoadapter.bench.second_backbone_summary import build_second_backbone_summary

    bad = _rows()
    bad[0] = dict(bad[0])
    bad[0].pop("macro_f1")

    with pytest.raises(ValueError, match="missing required fields"):
        build_second_backbone_summary(bad)


def test_write_second_backbone_summary_round_trips(tmp_path: Path):
    from geoadapter.bench.second_backbone_summary import write_second_backbone_summary

    raw_path = tmp_path / "raw.json"
    summary_path = tmp_path / "summary.json"
    raw_path.write_text(json.dumps(_rows()), encoding="utf-8")

    summary = write_second_backbone_summary(raw_path, summary_path)

    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded == summary
    assert loaded["row_count"] == 18
