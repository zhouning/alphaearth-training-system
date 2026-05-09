#!/usr/bin/env python
"""Generate LandCover.ai segmentation figure from paper12_results/landcoverai_segmentation.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper12_results" / "landcoverai_segmentation.json"
OUT = ROOT / "paper12" / "figures" / "landcoverai_segmentation.pdf"

with SRC.open("r", encoding="utf-8") as f:
    data = json.load(f)

by_method = defaultdict(list)
for r in data:
    by_method[r["method"]].append(r["mIoU"])

order = ["linear_probe", "lora_r8", "houlsby"]
labels = ["Linear Probe", "LoRA (r=8)", "Houlsby"]
colors = ["C4", "C3", "C2"]

means = [mean(by_method[m]) for m in order]
stds = [stdev(by_method[m]) if len(by_method[m]) > 1 else 0.0 for m in order]

fig, ax = plt.subplots(figsize=(5.0, 3.6))
bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="black", lw=0.6)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("mIoU")
ax.set_title("LandCover.ai Segmentation (6 classes, RGB)")
ax.set_ylim(0, 0.78)
ax.grid(True, axis="y", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
