"""Measure Prithvi + PEFT throughput on this machine.

Runs 50 forward + backward steps with realistic batch (8 patches @ 128×128 RGB),
times the average step, and projects training budget for the full Linhe demo
(170 scene * ~700 patches/scene * 20 epoch).

Usage:
  python scripts/bench_prithvi_throughput.py
  python scripts/bench_prithvi_throughput.py --batch-size 16 --steps 30
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--methods", nargs="*",
                   default=["linear_probe", "houlsby"],
                   help="subset of {linear_probe, houlsby, lora, bitfit, full_finetune}")
    args = p.parse_args()

    from geoadapter.models.prithvi import PrithviBackbone
    from geoadapter.models.heads import SegmentationHead
    from geoadapter.adapters import ZeroPadAdapter
    from geoadapter.adapters.lora import inject_lora
    from geoadapter.adapters.bitfit import configure_bitfit
    from geoadapter.adapters.houlsby import inject_houlsby_adapters

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}, batch={args.batch_size}, patch={args.patch}, steps={args.steps}")

    rows = []
    for method in args.methods:
        torch.manual_seed(42)
        backbone = PrithviBackbone(pretrained=False)
        if method == "lora":
            for blk in backbone.blocks:
                inject_lora(blk, rank=8)
        elif method == "bitfit":
            configure_bitfit(backbone)
        elif method == "houlsby":
            for blk in backbone.blocks:
                inject_houlsby_adapters(blk, bottleneck_dim=64)
        elif method == "full_finetune":
            for p_ in backbone.parameters():
                p_.requires_grad_(True)

        adapter = ZeroPadAdapter(in_channels=3, out_channels=6)
        head = SegmentationHead(in_dim=768, num_classes=2, patch_size=16)

        backbone.to(device); adapter.to(device); head.to(device)
        trainable = [p_ for p_ in list(adapter.parameters()) +
                     list(head.parameters()) +
                     list(backbone.parameters()) if p_.requires_grad]
        n_params = sum(p_.numel() for p_ in trainable)
        opt = torch.optim.Adam(trainable, lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.randn(args.batch_size, 3, args.patch, args.patch, device=device)
        y = torch.randint(0, 2, (args.batch_size, args.patch, args.patch), device=device)

        # warmup
        for _ in range(3):
            opt.zero_grad()
            feat, sd = backbone(adapter(x), return_spatial=True)
            logits = head(feat, sd)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.steps):
            opt.zero_grad()
            feat, sd = backbone(adapter(x), return_spatial=True)
            logits = head(feat, sd)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        per_step = dt / args.steps
        per_patch_ms = per_step / args.batch_size * 1000

        rows.append({
            "method": method,
            "trainable_params": n_params,
            "step_s": round(per_step, 3),
            "patch_ms": round(per_patch_ms, 1),
        })
        print(f"[ok] {method:14s} step={per_step*1000:.0f}ms ({per_patch_ms:.1f} ms/patch), trainable={n_params:,}")

    # Project full training budget
    print()
    print("=== Full Linhe demo projection (170 scenes × 700 patches × 20 epoch = 2.38M sample-passes) ===")
    samples = 170 * 700 * 20
    for r in rows:
        total_s = samples * r["patch_ms"] / 1000
        print(f"  {r['method']:14s}: {total_s/3600:.1f} h  ({total_s/60:.0f} min)")
    print()
    print("=== 2025Q1-Q4 demo subset (68 scenes × 700 patches × 20 epoch = 952K sample-passes) ===")
    samples = 68 * 700 * 20
    for r in rows:
        total_s = samples * r["patch_ms"] / 1000
        print(f"  {r['method']:14s}: {total_s/3600:.1f} h  ({total_s/60:.0f} min)")


if __name__ == "__main__":
    main()
