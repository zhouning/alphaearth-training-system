"""Generate synthetic binary masks for Linhe RGB patches — pipeline smoke test only.

Creates a per-patch mask npz where mask = (mean(RGB) >= threshold). Saves an
osm-compatible _osm_index.parquet so load_linhe_buildings() can pair patches
without OSM. NOT a real building label — purely to validate that
linhe_buildings -> Prithvi PEFT segmentation training converges before we
spend hours pulling OSM.

Outputs (under data/linhe_patches/_synth/):
  per-patch  mask_<row>_<col>.npz with key "mask" (uint8, 128x128)
  data/linhe_patches/_synth_index.parquet  with columns
      [patch_path, label_path, building_share]

Usage:
  python scripts/linhe_synth_masks.py                  # default threshold 140
  python scripts/linhe_synth_masks.py --threshold 120
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "linhe_patches"
INDEX = PATCH_DIR / "_index.parquet"
SYNTH_BASE = PATCH_DIR / "_synth"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=int, default=140,
                   help="mean(RGB) >= threshold → label 1")
    args = p.parse_args()

    if not INDEX.exists():
        raise SystemExit(f"missing {INDEX} — run linhe_build_patches.py first")

    idx = pd.read_parquet(INDEX)
    print(f"[info] {len(idx)} patches across {idx['scene_id'].nunique()} scenes")
    SYNTH_BASE.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, r in idx.iterrows():
        patch_path = ROOT / r["patch_path"]
        rgb = np.load(patch_path)["rgb"]  # CHW uint8
        mean_rgb = rgb.astype(np.float32).mean(axis=0)  # HW
        mask = (mean_rgb >= args.threshold).astype(np.uint8)
        scene_dir = SYNTH_BASE / r["scene_id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        out = scene_dir / f"m_{r['row']:05d}_{r['col']:05d}.npz"
        np.savez_compressed(out, mask=mask)
        rows.append({
            "patch_path": r["patch_path"],
            "label_path": str(out.relative_to(ROOT)),
            "building_share": float(mask.mean()),
        })
        if (i + 1) % 1000 == 0:
            print(f"[info] {i + 1}/{len(idx)}")

    out_idx = PATCH_DIR / "_synth_index.parquet"
    pd.DataFrame(rows).to_parquet(out_idx)
    pos = sum(r["building_share"] for r in rows) / len(rows)
    print(f"[ok] {len(rows)} synth masks → {out_idx}")
    print(f"[ok] avg positive share: {pos:.3f}  (threshold={args.threshold})")
    (SYNTH_BASE / "_manifest.json").write_text(json.dumps({
        "threshold": args.threshold,
        "n_patches": len(rows),
        "avg_positive_share": pos,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
