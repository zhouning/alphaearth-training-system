from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REQUIRED_FIELDS = {
    "backbone",
    "method",
    "modality",
    "seed",
    "trainable_params",
    "overall_accuracy",
    "macro_f1",
}


def _require_fields(row: dict[str, Any], index: int) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"row {index} missing required fields: {names}")


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def build_second_backbone_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        _require_fields(row, index)
        key = (str(row["backbone"]), str(row["method"]), str(row["modality"]))
        grouped[key].append(row)

    groups = []
    for (backbone, method, modality), group_rows in sorted(grouped.items()):
        oa = [float(row["overall_accuracy"]) for row in group_rows]
        f1 = [float(row["macro_f1"]) for row in group_rows]
        params = sorted({int(row["trainable_params"]) for row in group_rows})
        seeds = sorted(int(row["seed"]) for row in group_rows)
        groups.append(
            {
                "backbone": backbone,
                "method": method,
                "modality": modality,
                "trainable_params": params[0] if len(params) == 1 else params,
                "overall_accuracy_mean": mean(oa),
                "overall_accuracy_std": _std(oa),
                "macro_f1_mean": mean(f1),
                "macro_f1_std": _std(f1),
                "seeds": seeds,
            }
        )

    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_modality[group["modality"]].append(group)

    for modality_groups in by_modality.values():
        ranked = sorted(
            modality_groups,
            key=lambda item: item["overall_accuracy_mean"],
            reverse=True,
        )
        for rank, group in enumerate(ranked, start=1):
            group["rank_by_overall_accuracy"] = rank

    return {
        "schema": "paper12.second_backbone_eurosat_summary.v1",
        "row_count": len(rows),
        "groups": groups,
    }


def write_second_backbone_summary(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("second-backbone raw result file must contain a JSON list")
    summary = build_second_backbone_summary(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_second_backbone_summary(args.input, args.output)


if __name__ == "__main__":
    main()
