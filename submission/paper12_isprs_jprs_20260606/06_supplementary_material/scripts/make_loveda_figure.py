#!/usr/bin/env python
"""Generate LoveDA cross-domain 5-method bar figure for paper12 Section 9.4.

Reads results/loveda/loveda_lulc_seg.json (canonical merged file with one
direction-tagged row per (method, seed)). Renders two side-by-side panels
sharing a y-axis: U->R (Urban->Rural) and R->U (Rural->Urban).

Each panel draws bars for the 5 PEFT methods (mean +/- std over 3 seeds)
plus a horizontal dashed line at the all-class-0 majority-class baseline
(approximately 0.086 / 0.095 for U->R / R->U respectively, computed from
LoveDA validation class shares so the floor is empirical, not assumed).
Houlsby gets a "(+/-X pp)" callout above its bar to make the
capacity-threshold finding visible at a glance.

Methods missing from the JSON (e.g., before the diag sweep is folded in)
render as hatched empty bars labelled "TBD" so the figure remains usable
during paper drafting.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "loveda" / "loveda_lulc_seg.json"
OUT = ROOT / "paper12" / "figures" / "loveda_crossdomain.pdf"

ORDER = ["linear_probe", "bitfit", "lora_r8", "houlsby", "geoadapter"]
LABELS = ["Linear\nProbe", "BitFit", "LoRA\n(r=8)", "Houlsby", "Geo-\nAdapter"]
COLORS = ["#94A3B8", "#FBBF24", "#F87171", "#3B82F6", "#A855F7"]

DIRECTIONS = [("U->R", "Urban -> Rural"), ("R->U", "Rural -> Urban")]

BG_FLOOR = {"U->R": 0.0858, "R->U": 0.0952}


def load_records() -> list[dict]:
    if not SRC.exists():
        return []
    with SRC.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_by_direction(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    by_dir: dict[str, dict[str, list[float]]] = {d: defaultdict(list) for d, _ in DIRECTIONS}
    for r in records:
        d = r.get("direction")
        if d in by_dir and "mIoU" in r:
            by_dir[d][r["method"]].append(r["mIoU"])
    return by_dir


def draw_panel(ax, by_method, direction, title) -> None:
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

    x = list(range(len(ORDER)))
    plotted_means = [(m if m is not None else 0.05) for m in means]
    hatches = ["" if mn is not None else "//" for mn in means]
    alphas = [1.0 if mn is not None else 0.35 for mn in means]

    for xi, mn, sd, color, hatch, alpha in zip(
        x, plotted_means, stds, COLORS, hatches, alphas
    ):
        ax.bar(xi, mn, yerr=sd if mn > 0.05 else 0.0,
               capsize=4, color=color, edgecolor="black", lw=0.6,
               hatch=hatch, alpha=alpha)

    linear_mean = means[ORDER.index("linear_probe")]
    for xi, m_name, mn in zip(x, ORDER, means):
        if mn is None:
            ax.text(xi, 0.07, "TBD", ha="center", va="bottom",
                    fontsize=9, color="#475569", fontweight="bold")
            continue
        label = f"{mn:.3f}"
        if m_name == "houlsby" and linear_mean is not None:
            gain_pp = (mn - linear_mean) * 100
            label += f"\n(+{gain_pp:.1f}pp)"
        ax.text(xi, mn + 0.005, label, ha="center", va="bottom", fontsize=8)

    floor = BG_FLOOR[direction]
    ax.axhline(floor, color="#64748B", ls="--", lw=0.7)
    ax.text(len(ORDER) - 0.5, floor - 0.012,
            f"all-class-0 floor ({floor:.3f})",
            ha="right", va="top", fontsize=7, color="#64748B")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", ls=":", alpha=0.5)


def main() -> None:
    records = load_records()
    by_dir = aggregate_by_direction(records)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    for ax, (d, title) in zip(axes, DIRECTIONS):
        draw_panel(ax, by_dir[d], d, f"{title}  (n=3 seeds)")

    axes[0].set_ylabel("mIoU")

    all_means = []
    for d, _ in DIRECTIONS:
        for vals in by_dir[d].values():
            if vals:
                all_means.append(mean(vals))
    ymax = max(all_means + [0.25])
    for ax in axes:
        ax.set_ylim(0, max(0.30, ymax + 0.05))

    fig.suptitle(
        "LoveDA cross-domain transfer: 7-class semantic segmentation (RGB, Prithvi-100M frozen)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    filled = sum(
        1 for d, _ in DIRECTIONS for m in ORDER if by_dir[d].get(m)
    )
    print(f"method-direction cells filled: {filled}/{2 * len(ORDER)}")


if __name__ == "__main__":
    main()
