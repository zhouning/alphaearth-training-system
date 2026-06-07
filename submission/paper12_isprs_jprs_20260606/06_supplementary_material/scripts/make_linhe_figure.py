#!/usr/bin/env python
"""Generate Linhe LULC 5-method bar figure for paper12 Section 9.

Reads linhe_results/linhe_lulc_seg.json. Each entry should contain:
    {"method": str, "modality": "rgb_3band", "seed": int,
     "trainable_params": int, "mIoU": float}

Methods missing from the JSON are rendered as hatched empty bars
labelled "TBD" so the figure is still usable for the demo before the
full 5-method Colab run is back.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "linhe_results" / "linhe_lulc_seg.json"
OUT = ROOT / "paper12" / "figures" / "linhe_lulc_seg.pdf"

ORDER = ["linear_probe", "bitfit", "lora_r8", "houlsby", "geoadapter"]
LABELS = ["Linear\nProbe", "BitFit", "LoRA\n(r=8)", "Houlsby", "Geo-\nAdapter"]
COLORS = ["#94A3B8", "#FBBF24", "#F87171", "#3B82F6", "#A855F7"]

PARAMS_FALLBACK = {
    "linear_probe": 4_614,
    "bitfit": 110_982,
    "lora_r8": 152_070,
    "houlsby": 1_194_246,
    "geoadapter": 8_214,
}


def load_records() -> list[dict]:
    if not SRC.exists():
        return []
    with SRC.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    records = load_records()
    by_method: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if "mIoU" in r:
            by_method[r["method"]].append(r["mIoU"])

    means: list[float | None] = []
    stds: list[float] = []
    for m in ORDER:
        vals = by_method.get(m, [])
        if vals:
            means.append(mean(vals))
            stds.append(stdev(vals) if len(vals) > 1 else 0.0)
        else:
            means.append(None)
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = list(range(len(ORDER)))
    plotted_means = [(m if m is not None else 0.05) for m in means]
    plotted_stds = [s if mn is not None else 0.0 for s, mn in zip(stds, means)]
    hatches = ["" if mn is not None else "//" for mn in means]
    alphas = [1.0 if mn is not None else 0.35 for mn in means]

    bars = []
    for xi, mn, sd, color, hatch, alpha in zip(
        x, plotted_means, plotted_stds, COLORS, hatches, alphas
    ):
        b = ax.bar(xi, mn, yerr=sd if mn > 0.05 else 0.0,
                   capsize=5, color=color, edgecolor="black", lw=0.6,
                   hatch=hatch, alpha=alpha)
        bars.append(b[0])

    for xi, mn in zip(x, means):
        if mn is None:
            ax.text(xi, 0.07, "TBD", ha="center", va="bottom",
                    fontsize=10, color="#475569", fontweight="bold")
        else:
            ax.text(xi, mn + 0.012, f"{mn:.3f}", ha="center", va="bottom",
                    fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("mIoU")
    ax.set_title("Linhe LULC 6-class (RGB, scene-level split, Esri 2022 labels)")
    ymax = max([m for m in means if m is not None] + [0.30])
    ax.set_ylim(0, max(0.45, ymax + 0.15))
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    if means[ORDER.index("linear_probe")] is not None:
        ax.axhline(1 / 6, color="#64748B", ls="--", lw=0.7)
        ax.text(len(ORDER) - 0.5, 1 / 6 + 0.005,
                "constant-pred floor (1/6)", ha="right", va="bottom",
                fontsize=7, color="#64748B")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    filled = sum(1 for m in means if m is not None)
    print(f"methods filled: {filled}/{len(ORDER)}")


if __name__ == "__main__":
    main()
