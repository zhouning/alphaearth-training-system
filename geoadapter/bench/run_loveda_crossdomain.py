"""Orchestrate LoveDA cross-domain runs for paper12 Section 9.4.

Runs the U->R and R->U configs sequentially via run_benchmark.main, then merges
the two flat result JSONs into a single file with a `direction` field per row.

Usage:
    python -m geoadapter.bench.run_loveda_crossdomain \\
        --u2r-config geoadapter/bench/configs/loveda_lulc_u2r.yaml \\
        --r2u-config geoadapter/bench/configs/loveda_lulc_r2u.yaml \\
        --output results/loveda/loveda_lulc_seg.json \\
        --checkpoint-dir checkpoints/loveda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def merge_direction_results(u2r_path: Path, r2u_path: Path, out_path: Path) -> None:
    """Merge two flat JSON arrays into one, tagging each row with `direction`."""
    u2r_path = Path(u2r_path)
    r2u_path = Path(r2u_path)
    out_path = Path(out_path)
    if not u2r_path.exists():
        raise FileNotFoundError(f"U->R results missing: {u2r_path}")
    if not r2u_path.exists():
        raise FileNotFoundError(f"R->U results missing: {r2u_path}")
    merged = []
    for row in json.loads(u2r_path.read_text(encoding="utf-8")):
        merged.append({**row, "direction": "U->R"})
    for row in json.loads(r2u_path.read_text(encoding="utf-8")):
        merged.append({**row, "direction": "R->U"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged {len(merged)} rows -> {out_path}")


def _run_one_direction(config: Path, output: Path, checkpoint_dir: Path | None) -> None:
    cmd = [sys.executable, "-m", "geoadapter.bench.run_benchmark",
           "--config", str(config), "--output", str(output)]
    if checkpoint_dir is not None:
        cmd += ["--checkpoint-dir", str(checkpoint_dir)]
    print(f"\n=== Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u2r-config", required=True, type=Path)
    parser.add_argument("--r2u-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Final merged JSON for paper figure (e.g. results/loveda/loveda_lulc_seg.json)")
    parser.add_argument("--checkpoint-dir", default=None, type=Path,
                        help="Directory for per-epoch checkpoints (per direction subdir)")
    parser.add_argument("--skip-runs", action="store_true",
                        help="Skip training subprocesses and only merge existing per-direction JSONs")
    args = parser.parse_args()

    out_dir = args.output.parent
    stem = args.output.stem
    u2r_out = out_dir / f"{stem}_u2r.json"
    r2u_out = out_dir / f"{stem}_r2u.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_runs:
        ckpt_u2r = (args.checkpoint_dir / "u2r") if args.checkpoint_dir else None
        ckpt_r2u = (args.checkpoint_dir / "r2u") if args.checkpoint_dir else None
        _run_one_direction(args.u2r_config, u2r_out, ckpt_u2r)
        _run_one_direction(args.r2u_config, r2u_out, ckpt_r2u)

    merge_direction_results(u2r_out, r2u_out, args.output)


if __name__ == "__main__":
    main()
