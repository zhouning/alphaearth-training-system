#!/usr/bin/env python
"""Generate LoRA rank sensitivity figure from paper12_results/lora_rank_ablation.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper12_results" / "lora_rank_ablation.json"
OUT = ROOT / "paper12" / "figures" / "lora_rank_sensitivity.pdf"

# Map observed trainable parameter counts back to rank.
PARAM_TO_RANK = {
    302602: 4,
    597514: 8,
    1187338: 16,
    2366986: 32,
}

with SRC.open("r", encoding="utf-8") as f:
    data = json.load(f)

rows = [r for r in data if r["method"] == "lora_split_qkv" and r["modality"] == "s2_full"]
by_rank = defaultdict(list)
for r in rows:
    rank = PARAM_TO_RANK.get(r["trainable_params"])
    if rank is not None:
        by_rank[rank].append(r["overall_accuracy"])

ranks = sorted(by_rank)
means = [mean(by_rank[r]) for r in ranks]
stds = [stdev(by_rank[r]) if len(by_rank[r]) > 1 else 0.0 for r in ranks]

fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.errorbar(ranks, means, yerr=stds, fmt="o-", capsize=4, color="C3", lw=2, label="LoRA Split-QKV")
ax.axhline(0.821, color="C2", ls="--", lw=1.5, label="Houlsby")
ax.axhline(0.702, color="C1", ls=":", lw=1.5, label="BitFit")
ax.axhline(0.657, color="C4", ls=":", lw=1.5, alpha=0.8, label="Linear Probe")
ax.set_xlabel("LoRA rank r")
ax.set_ylabel("EuroSAT OA (s2_full)")
ax.set_xticks(ranks)
ax.set_ylim(0.65, 0.84)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
