import numpy as np
import pytest

from geoadapter.bench.geovlm_prompt_summary import (
    binary_metrics,
    build_summary,
    paired_bootstrap_delta,
)


def _passing_rows():
    rows = []
    for seed in (42, 123, 456):
        for name, iou in (("building", 0.50), ("road", 0.40), ("water", 0.45)):
            rows.append(
                {
                    "method": "siglip_film_dense_similarity_houlsby",
                    "seed": seed,
                    "class_name": name,
                    "seen_iou": iou,
                    "held_out_iou": iou * 0.95,
                    "correct_iou_by_sample": [iou, iou + 0.02, iou - 0.01],
                    "wrong_iou_by_sample": [
                        iou - 0.20,
                        iou - 0.18,
                        iou - 0.22,
                    ],
                    "prompt_probability_change_by_sample": [0.08, 0.09, 0.07],
                    "checkpoint_reproduced": True,
                }
            )
            rows.append(
                {
                    "method": "no_text_three_binary_heads_houlsby",
                    "seed": seed,
                    "class_name": name,
                    "checkpoint_reproduced": True,
                }
            )
    return rows


def test_binary_metrics_treats_two_empty_masks_as_perfect():
    metrics = binary_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
    assert metrics == {"foreground_iou": 1.0, "dice": 1.0}


def test_paired_bootstrap_is_deterministic_and_positive():
    first = paired_bootstrap_delta([0.5, 0.6], [0.2, 0.3], iterations=100, seed=9)
    second = paired_bootstrap_delta([0.5, 0.6], [0.2, 0.3], iterations=100, seed=9)
    assert first == second
    assert first["mean_delta"] == pytest.approx(0.3)
    assert first["ci95_low"] > 0


def test_summary_passes_all_confirmed_mvp_gates():
    summary = build_summary(_passing_rows(), bootstrap_iterations=1000, seed=7)
    assert summary["schema"] == "paper12.geovlm_prompt_summary.v1"
    assert summary["mvp_status"] == "passed"
    assert summary["failed_gates"] == []


def test_summary_reports_failed_class_gate():
    rows = _passing_rows()
    for row in rows:
        if row["class_name"] == "road":
            row["seen_iou"] = 0.20
            row["held_out_iou"] = 0.19
    summary = build_summary(rows, bootstrap_iterations=100, seed=7)
    assert summary["mvp_status"] == "failed"
    assert "class_iou:road<0.25" in summary["failed_gates"]


def test_summary_requires_the_baseline_method_seed_matrix():
    rows = [
        row
        for row in _passing_rows()
        if row["method"] == "siglip_film_dense_similarity_houlsby"
    ]

    summary = build_summary(rows, bootstrap_iterations=100, seed=7)

    assert summary["mvp_status"] == "incomplete"
    assert summary["incomplete_reasons"] == [
        "missing_method_seed:no_text_three_binary_heads_houlsby:42",
        "missing_method_seed:no_text_three_binary_heads_houlsby:123",
        "missing_method_seed:no_text_three_binary_heads_houlsby:456",
    ]


def test_summary_fails_when_any_checkpoint_does_not_reproduce():
    rows = _passing_rows()
    rows[0]["checkpoint_reproduced"] = False

    summary = build_summary(rows, bootstrap_iterations=100, seed=7)

    assert summary["mvp_status"] == "failed"
    assert summary["gates"]["checkpoint_reproduction"] is False
    assert "checkpoint_reproduction_failed" in summary["failed_gates"]


def test_summary_marks_single_seed_stage_incomplete():
    rows = [row for row in _passing_rows() if row["seed"] == 42]
    summary = build_summary(rows, bootstrap_iterations=100, seed=7)
    assert summary["mvp_status"] == "incomplete"
    assert summary["incomplete_reasons"] == [
        "missing_method_seed:siglip_film_dense_similarity_houlsby:123",
        "missing_method_seed:siglip_film_dense_similarity_houlsby:456",
        "missing_method_seed:no_text_three_binary_heads_houlsby:123",
        "missing_method_seed:no_text_three_binary_heads_houlsby:456",
    ]


def test_summary_rejects_duplicate_and_synthetic_rows():
    rows = _passing_rows()
    with pytest.raises(ValueError, match="duplicate"):
        build_summary(rows + [dict(rows[0])], bootstrap_iterations=10)

    rows = _passing_rows()
    rows[0]["synthetic_fallback"] = True
    with pytest.raises(ValueError, match="synthetic"):
        build_summary(rows, bootstrap_iterations=10)
