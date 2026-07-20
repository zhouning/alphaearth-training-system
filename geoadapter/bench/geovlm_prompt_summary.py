from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROMPT_METHOD = "siglip_film_dense_similarity_houlsby"
REQUIRED_SEEDS = (42, 123, 456)
REQUIRED_CLASSES = ("building", "road", "water")


def binary_metrics(target, prediction):
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical shapes")
    intersection = np.logical_and(target, prediction).sum()
    union = np.logical_or(target, prediction).sum()
    iou = 1.0 if union == 0 else float(intersection / union)
    denom = target.sum() + prediction.sum()
    dice = 1.0 if denom == 0 else float(2 * intersection / denom)
    return {"foreground_iou": iou, "dice": dice}


def paired_bootstrap_delta(correct, wrong, *, iterations=1000, seed=0):
    correct = np.asarray(correct, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    if correct.shape != wrong.shape or correct.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not np.isfinite(correct).all() or not np.isfinite(wrong).all():
        raise ValueError("paired bootstrap inputs must be finite")
    deltas = correct - wrong
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        draw = rng.integers(0, deltas.size, size=deltas.size)
        samples[index] = deltas[draw].mean()
    return {
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_keys: set[tuple[str, int, str]] = set()
    prompt_rows = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("result rows must be mappings")
        if raw.get("synthetic_fallback") is True:
            raise ValueError("synthetic fallback rows are forbidden")
        method = str(raw.get("method", ""))
        seed = int(raw.get("seed", -1))
        class_name = str(raw.get("class_name", ""))
        if seed not in REQUIRED_SEEDS:
            raise ValueError(f"unsupported seed: {seed}")
        if class_name not in REQUIRED_CLASSES:
            raise ValueError(f"unsupported class_name: {class_name}")
        key = (method, seed, class_name)
        if key in seen_keys:
            raise ValueError(f"duplicate result row: {key}")
        seen_keys.add(key)
        if method != PROMPT_METHOD:
            continue

        row = dict(raw)
        row["seen_iou"] = _finite_float(raw.get("seen_iou"), "seen_iou")
        row["held_out_iou"] = _finite_float(
            raw.get("held_out_iou"), "held_out_iou"
        )
        arrays = []
        for label in (
            "correct_iou_by_sample",
            "wrong_iou_by_sample",
            "prompt_probability_change_by_sample",
        ):
            values = np.asarray(raw.get(label, ()), dtype=float)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"{label} must be a non-empty finite vector")
            arrays.append(values)
            row[label] = values
        if len({array.size for array in arrays}) != 1:
            raise ValueError("per-sample arrays must be aligned")
        prompt_rows.append(row)
    return prompt_rows


def _incomplete_reasons(rows: Sequence[dict[str, Any]]) -> list[str]:
    keys = {(int(row["seed"]), str(row["class_name"])) for row in rows}
    present_seeds = {seed for seed, _ in keys}
    present_classes = {class_name for _, class_name in keys}
    reasons = [
        f"missing_seed:{seed}" for seed in REQUIRED_SEEDS if seed not in present_seeds
    ]
    reasons.extend(
        f"missing_class:{class_name}"
        for class_name in REQUIRED_CLASSES
        if class_name not in present_classes
    )
    for seed in REQUIRED_SEEDS:
        for class_name in REQUIRED_CLASSES:
            if (
                seed in present_seeds
                and class_name in present_classes
                and (seed, class_name) not in keys
            ):
                reasons.append(f"missing_row:{seed}:{class_name}")
    return reasons


def build_summary(rows, *, bootstrap_iterations=1000, seed=0):
    prompt_rows = _validate_rows(list(rows))
    incomplete = _incomplete_reasons(prompt_rows)
    base = {
        "schema": "paper12.geovlm_prompt_summary.v1",
        "method": PROMPT_METHOD,
        "required_seeds": list(REQUIRED_SEEDS),
        "required_classes": list(REQUIRED_CLASSES),
    }
    if incomplete:
        return {
            **base,
            "mvp_status": "incomplete",
            "incomplete_reasons": incomplete,
            "failed_gates": [],
            "gates": {},
            "metrics": {},
        }

    class_iou = {
        class_name: float(
            np.mean(
                [
                    row["seen_iou"]
                    for row in prompt_rows
                    if row["class_name"] == class_name
                ]
            )
        )
        for class_name in REQUIRED_CLASSES
    }
    seen_iou = float(np.mean([row["seen_iou"] for row in prompt_rows]))
    held_out_iou = float(np.mean([row["held_out_iou"] for row in prompt_rows]))
    correct = np.concatenate(
        [row["correct_iou_by_sample"] for row in prompt_rows]
    )
    wrong = np.concatenate([row["wrong_iou_by_sample"] for row in prompt_rows])
    probability_change = float(
        np.mean(
            np.concatenate(
                [row["prompt_probability_change_by_sample"] for row in prompt_rows]
            )
        )
    )
    bootstrap = paired_bootstrap_delta(
        correct,
        wrong,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    gates = {
        "mean_foreground_iou": seen_iou >= 0.40,
        "each_class_iou": all(value >= 0.25 for value in class_iou.values()),
        "held_out_retention": held_out_iou >= 0.90 * seen_iou,
        "correct_minus_wrong_iou": bootstrap["mean_delta"] >= 0.10,
        "counterfactual_ci_positive": bootstrap["ci95_low"] > 0.0,
        "prompt_probability_change": probability_change >= 0.05,
    }
    failed = []
    if not gates["mean_foreground_iou"]:
        failed.append("mean_foreground_iou<0.40")
    for class_name, value in class_iou.items():
        if value < 0.25:
            failed.append(f"class_iou:{class_name}<0.25")
    if not gates["held_out_retention"]:
        failed.append("held_out_retention<0.90")
    if not gates["correct_minus_wrong_iou"]:
        failed.append("correct_minus_wrong_iou<0.10")
    if not gates["counterfactual_ci_positive"]:
        failed.append("counterfactual_ci95_low<=0")
    if not gates["prompt_probability_change"]:
        failed.append("prompt_probability_change<0.05")
    return {
        **base,
        "mvp_status": "failed" if failed else "passed",
        "incomplete_reasons": [],
        "failed_gates": failed,
        "gates": gates,
        "metrics": {
            "mean_seen_iou": seen_iou,
            "mean_held_out_iou": held_out_iou,
            "class_seen_iou": class_iou,
            "paired_bootstrap": bootstrap,
            "mean_prompt_probability_change": probability_change,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Paper12 GeoVLM gate summary")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload.get("rows", ()) if isinstance(payload, dict) else payload
    summary = build_summary(
        rows,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
