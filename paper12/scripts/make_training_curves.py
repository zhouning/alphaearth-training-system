#!/usr/bin/env python
"""Parse training logs and generate loss-curve figures.

Outputs:
    paper12/figures/training_curves_eurosat.pdf
    paper12/figures/training_curves_bigearthnet.pdf

Usage:
    python paper12/scripts/make_training_curves.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "paper12_results"
FIG_DIR = ROOT / "paper12" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_RE = re.compile(
    r"\[(?P<method>[^|]+)\|(?P<mod>[^|]+)\|seed=(?P<seed>\d+)\]\s+"
    r"Epoch\s+(?P<epoch>\d+)/\d+\s+loss=(?P<loss>[\d.]+)"
)

RESULT_RE = re.compile(
    r"\[(?P<method>[^|]+)\|(?P<mod>[^|]+)\|seed=(?P<seed>\d+)\]\s+"
    r"OA=(?P<oa>[\d.]+)"
)


def parse_logs(pattern: str):
    curves = defaultdict(lambda: defaultdict(list))
    for p in sorted(LOG_DIR.glob(pattern)):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = EPOCH_RE.search(line)
            if m:
                key = (m.group("method"), m.group("mod"), m.group("seed"))
                curves[key]["epoch"].append(int(m.group("epoch")))
                curves[key]["loss"].append(float(m.group("loss")))
    return curves


def plot_eurosat():
    curves = {}
    for pat in ["full_finetune_*.log", "lora_ablation_*.log"]:
        curves.update(parse_logs(pat))

    # EuroSAT main methods from the original experiment logs
    # The main eurosat logs are in results/ dir but we don't have them as .log
    # We have full_finetune and lora_ablation logs — plot those as they show
    # the most interesting contrast (full FT vs split-QKV on s2_full)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    styles = {
        "full_finetune": ("Full Fine-Tuning", "C0", "-"),
        "lora_split_qkv_r8": ("LoRA Split-QKV", "C3", "--"),
    }

    plotted = set()
    for (method, mod, seed), data in sorted(curves.items()):
        if mod != "s2_full" or seed != "42":
            continue
        if method not in styles:
            continue
        label, color, ls = styles[method]
        ax.plot(data["epoch"], data["loss"], color=color, ls=ls,
                marker="o", markersize=4, label=label)
        plotted.add(method)

    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title("EuroSAT s2\\_full (seed=42)")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        out = FIG_DIR / "training_curves_eurosat.pdf"
        fig.savefig(out)
        print(f"wrote {out}")
    else:
        print("no EuroSAT curves found")
    plt.close(fig)


def plot_bigearthnet():
    curves = {}
    for pat in ["bigearthnet_*.log"]:
        curves.update(parse_logs(pat))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    styles = {
        "houlsby": ("Houlsby", "C2", "-"),
        "linear_probe": ("Linear Probe", "C4", "--"),
        "lora_r8": ("LoRA (r=8)", "C1", "-."),
    }

    plotted = set()
    for (method, mod, seed), data in sorted(curves.items()):
        if mod != "s2_full" or seed != "42":
            continue
        if method not in styles:
            continue
        label, color, ls = styles[method]
        ax.plot(data["epoch"], data["loss"], color=color, ls=ls,
                marker="o", markersize=4, label=label)
        plotted.add(method)

    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title("BigEarthNet-S2 s2\\_full (seed=42)")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        out = FIG_DIR / "training_curves_bigearthnet.pdf"
        fig.savefig(out)
        print(f"wrote {out}")
    else:
        print("no BigEarthNet curves found")
    plt.close(fig)


if __name__ == "__main__":
    plot_eurosat()
    plot_bigearthnet()
