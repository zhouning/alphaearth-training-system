#!/usr/bin/env python
"""Generate paper figures from experiment result JSON files.

Outputs:
    paper12/figures/acc_vs_params.pdf
    paper12/figures/per_modality_oa.pdf

Usage:
    python paper12/scripts/make_figures.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EUROSAT = ROOT / "results" / "eurosat_results.json"
FIG_DIR = ROOT / "paper12" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Hand-collated values for entries not in eurosat_results.json
FULL_FT_S2 = (0.869, 86_244_874)
SPLIT_QKV_S2 = (0.707, 597_514)  # mean of 3 seeds
SPLIT_QKV_RGB = (0.498, 597_514)


def load_eurosat():
    with EUROSAT.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(records, method, modality):
    xs = [r["overall_accuracy"] for r in records
          if r["method"] == method and r["modality"] == modality]
    params = next((r["trainable_params"] for r in records
                   if r["method"] == method and r["modality"] == modality), None)
    if not xs:
        return None
    return mean(xs), (stdev(xs) if len(xs) > 1 else 0.0), params


def fig_acc_vs_params(records):
    points = []
    for method, label in [
        ("linear_probe", "Linear Probe"),
        ("bitfit", "BitFit"),
        ("lora_r8", "LoRA (r=8)"),
        ("houlsby", "Houlsby"),
        ("geoadapter", "GeoAdapter v2"),
    ]:
        agg = aggregate(records, method, "s2_full")
        if agg:
            m, s, p = agg
            points.append((label, p, m, s))
    points.append(("LoRA Split-QKV", SPLIT_QKV_S2[1], SPLIT_QKV_S2[0], 0.003))
    points.append(("Full Fine-Tuning", FULL_FT_S2[1], FULL_FT_S2[0], 0.0))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, p, m, s in points:
        ax.errorbar(p, m, yerr=s, fmt="o", markersize=7, capsize=3)
        ax.annotate(label, (p, m), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("EuroSAT overall accuracy (s2\\_full)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_ylim(0.35, 0.92)
    fig.tight_layout()
    out = FIG_DIR / "acc_vs_params.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


def fig_per_modality(records):
    methods = [
        ("linear_probe", "Linear Probe"),
        ("bitfit", "BitFit"),
        ("lora_r8", "LoRA (r=8)"),
        ("houlsby", "Houlsby"),
        ("geoadapter", "GeoAdapter v2"),
    ]
    modalities = ["s2_full", "gf2", "rgb_sar", "rgb", "sar_only"]
    data = defaultdict(dict)
    for method, _ in methods:
        for mod in modalities:
            agg = aggregate(records, method, mod)
            if agg:
                data[method][mod] = agg

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    n_mod = len(modalities)
    bar_w = 0.15
    xs = list(range(len(methods)))
    for i, mod in enumerate(modalities):
        means, errs = [], []
        for method, _ in methods:
            if mod in data[method]:
                m, s, _ = data[method][mod]
            else:
                m, s = 0, 0
            means.append(m)
            errs.append(s)
        offset = (i - (n_mod - 1) / 2) * bar_w
        ax.bar([x + offset for x in xs], means, bar_w,
               yerr=errs, capsize=2, label=mod)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[1] for m in methods])
    ax.set_ylabel("EuroSAT overall accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(ncol=5, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 1.15), frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "per_modality_oa.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    records = load_eurosat()
    fig_acc_vs_params(records)
    fig_per_modality(records)


if __name__ == "__main__":
    main()
