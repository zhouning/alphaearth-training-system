"""Smoke-test Prithvi-100M on a Linhe RGB patch.

Verifies the end-to-end forward path that Phase 1.a/1.b will rely on:
  1. checkpoint loads from data/weights/prithvi/Prithvi_100M.pt
  2. 3-channel RGB → ZeroPadAdapter → 6-channel input
  3. CLS feature is [B, 768] and spatial tokens are [B, N, 768] with N = (patch/16)^2
  4. all backbone parameters are frozen (requires_grad=False)
  5. inference dtype matches input and there is no NaN/Inf

Reads:
  data/linhe_patches/_index.parquet  (built by linhe_build_patches.py)
  data/linhe_patches/<scene>/p_*.npz (CHW uint8 RGB tiles)
  data/weights/prithvi/Prithvi_100M.pt

Usage:
  python scripts/linhe_prithvi_smoke.py                 # auto-pick the first valid patch
  python scripts/linhe_prithvi_smoke.py --n-patches 4   # batch of 4
  python scripts/linhe_prithvi_smoke.py --device cuda   # if CUDA available
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geoadapter.adapters.zero_pad import ZeroPadAdapter
from geoadapter.models.prithvi import PrithviBackbone

ROOT = Path(__file__).resolve().parents[1]
PATCH_INDEX = ROOT / "data" / "linhe_patches" / "_index.parquet"
WEIGHTS = ROOT / "data" / "weights" / "prithvi" / "Prithvi_100M.pt"


def load_patches(n: int, patch_size_expected: int) -> tuple[torch.Tensor, list[str]]:
    """Load N RGB patches as a [N, 3, H, W] float tensor in [0, 1]."""
    if not PATCH_INDEX.exists():
        raise SystemExit(f"[error] {PATCH_INDEX} not found — run linhe_build_patches.py first")
    idx = pd.read_parquet(PATCH_INDEX)
    if len(idx) == 0:
        raise SystemExit("[error] _index.parquet is empty")

    chosen, used_paths = [], []
    for _, r in idx.iterrows():
        p = ROOT / r["patch_path"]
        if not p.exists():
            continue
        arr = np.load(p)["rgb"]  # CHW uint8
        if arr.ndim != 3 or arr.shape[0] != 3:
            continue
        if arr.shape[1] != patch_size_expected or arr.shape[2] != patch_size_expected:
            continue
        chosen.append(arr)
        used_paths.append(str(p.relative_to(ROOT)))
        if len(chosen) >= n:
            break

    if not chosen:
        raise SystemExit("[error] no RGB patch matched the expected size / shape")
    x = np.stack(chosen).astype(np.float32) / 255.0
    return torch.from_numpy(x), used_paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-patches", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=128, help="must match builder output")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--return-spatial", action="store_true", default=True,
                   help="also report spatial token tensor (default on)")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    # 1. patches
    print(f"[info] loading {args.n_patches} patches (expecting {args.patch_size}×{args.patch_size}) …")
    rgb, paths = load_patches(args.n_patches, args.patch_size)
    print(f"[ok]   tensor shape={tuple(rgb.shape)}, dtype={rgb.dtype}, "
          f"min={rgb.min():.3f}, max={rgb.max():.3f}")
    for q in paths:
        print(f"         · {q}")

    # 2. 3ch → 6ch via ZeroPad
    adapter = ZeroPadAdapter(in_channels=3, out_channels=6)
    x6 = adapter(rgb)
    print(f"[ok]   after ZeroPadAdapter: {tuple(x6.shape)} (channels 3-5 are zero)")
    assert (x6[:, 3:].abs().sum() == 0), "zero-pad channels should be exactly 0"

    # 3. backbone
    if not WEIGHTS.exists():
        raise SystemExit(f"[error] {WEIGHTS} not found")
    print(f"[info] loading Prithvi-100M from {WEIGHTS.relative_to(ROOT)} …")
    t0 = time.time()
    model = PrithviBackbone(pretrained=True, checkpoint_path=str(WEIGHTS))
    model.eval().to(args.device)
    x6 = x6.to(args.device)
    print(f"[ok]   model built in {time.time()-t0:.2f}s, device={args.device}")

    # 4. frozen check
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    total = sum(q.numel() for q in model.parameters())
    print(f"[ok]   backbone params: total={total/1e6:.2f}M, trainable={trainable} "
          f"({'frozen' if trainable == 0 else 'WARN: not frozen'})")
    assert trainable == 0, "backbone should be fully frozen"

    # 5. forward — CLS pooled
    with torch.no_grad():
        t0 = time.time()
        cls = model(x6)
        cls_ms = (time.time() - t0) * 1000
    print(f"[ok]   CLS feature: shape={tuple(cls.shape)}, dtype={cls.dtype}, "
          f"mean={cls.mean().item():+.4f}, std={cls.std().item():.4f}, "
          f"latency={cls_ms:.1f}ms")
    assert cls.shape == (args.n_patches, 768), f"expected [N,768], got {tuple(cls.shape)}"
    assert torch.isfinite(cls).all(), "CLS feature contains NaN/Inf"

    # 6. forward — spatial tokens (for segmentation head)
    with torch.no_grad():
        t0 = time.time()
        tokens, (h, w) = model(x6, return_spatial=True)
        seg_ms = (time.time() - t0) * 1000
    expected_n = (args.patch_size // 16) ** 2
    print(f"[ok]   spatial tokens: shape={tuple(tokens.shape)}, grid=({h},{w}), "
          f"expected_N={expected_n}, latency={seg_ms:.1f}ms")
    assert tokens.shape == (args.n_patches, expected_n, 768)
    assert (h, w) == (args.patch_size // 16, args.patch_size // 16)
    assert torch.isfinite(tokens).all(), "spatial tokens contain NaN/Inf"

    # 7. summary
    summary = {
        "device": args.device,
        "n_patches": args.n_patches,
        "patch_size": args.patch_size,
        "backbone_params_M": round(total / 1e6, 2),
        "backbone_trainable": int(trainable),
        "cls_shape": list(cls.shape),
        "spatial_token_shape": list(tokens.shape),
        "spatial_grid": [h, w],
        "cls_latency_ms": round(cls_ms, 1),
        "seg_latency_ms": round(seg_ms, 1),
        "patches_used": paths,
    }
    print("\n[summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[done] Prithvi smoke OK — Phase 1.0 forward path verified.")


if __name__ == "__main__":
    main()
