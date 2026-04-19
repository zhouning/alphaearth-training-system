"""Run Linhe real-label benchmark: linear_probe / bitfit / houlsby.

Calls linhe_finetune.py three times with identical hyperparams,
then aggregates results into benchmark_real_labels.{json,csv}.

Usage:
  python scripts/linhe_benchmark.py --epochs 10 --batch-size 16
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "linhe"
METHODS = ["linear_probe", "bitfit", "houlsby"]
PYTHON = sys.executable


def run_one(method: str, extra_args: list[str]) -> dict:
    cmd = [
        PYTHON, str(ROOT / "scripts" / "linhe_finetune.py"),
        "--peft-method", method,
        *extra_args,
    ]
    print(f"\n{'='*60}")
    print(f"  Running: {method}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ERROR] {method} failed with exit code {result.returncode}")
        return {}
    report_path = OUT / f"linhe_{method}_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    extra = sys.argv[1:]
    results = []
    for method in METHODS:
        r = run_one(method, extra)
        if r:
            results.append(r)

    if not results:
        print("[ERROR] no results collected")
        return

    summary_path = OUT / "benchmark_real_labels.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for r in results:
        row = {
            "method": r["peft_method"],
            "best_val_acc": r["best_val_acc"],
            "weighted_f1": r["weighted_f1"],
            "macro_f1": r["macro_f1"],
            "trainable_params": r["trainable_params"],
        }
        for cls, f1 in r.get("per_class_f1", {}).items():
            row[f"f1_{cls}"] = f1
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUT / "benchmark_real_labels.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("  Benchmark Summary")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\n[done] {summary_path}")
    print(f"[done] {csv_path}")


if __name__ == "__main__":
    main()
