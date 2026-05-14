"""Package Linhe data subset for Colab upload.

Produces a single tar.gz under packaging/colab/ with:
  data/linhe_patches/{scene}/p_*.npz       (RGB patches, filtered by quarter)
  data/linhe_patches/_index.parquet         (subset index)
  data/linhe_patches/_synth/{scene}/m_*.npz (synth masks)
  data/linhe_patches/_osm_index.parquet     (synth-as-OSM alias index)

Plus a sibling .sha256 file with the archive checksum so Colab can verify.

Usage:
  python scripts/package_colab_data.py --quarters 2025Q1 2025Q2 2025Q3 2025Q4
  python scripts/package_colab_data.py --quarters 2025Q3 2025Q4 --out linhe_2025_h2.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "linhe_patches"
OUT_DIR = ROOT / "packaging" / "colab"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quarters", nargs="+", required=True,
                   help="quarters to include, e.g. 2025Q1 2025Q2")
    p.add_argument("--out", default=None,
                   help="output filename (default: linhe_<q1>_<qn>.tar.gz)")
    args = p.parse_args()

    idx_path = PATCH_DIR / "_index.parquet"
    osm_path = PATCH_DIR / "_osm_index.parquet"
    if not idx_path.exists():
        raise SystemExit(f"missing {idx_path} — run linhe_build_patches.py first")
    if not osm_path.exists():
        raise SystemExit(f"missing {osm_path} — run linhe_synth_masks.py "
                         "(or linhe_rasterize_buildings.py) first")

    idx = pd.read_parquet(idx_path)
    sub = idx[idx["quarter"].isin(args.quarters)].copy()
    if sub.empty:
        raise SystemExit(f"no patches in quarters {args.quarters}")
    print(f"[info] {len(sub)} patches across {sub['scene_id'].nunique()} scenes")

    osm = pd.read_parquet(osm_path)
    # Schema differs: synth_masks → label_path; rasterize_buildings → osm_path
    if "label_path" not in osm.columns and "osm_path" in osm.columns:
        osm = osm.rename(columns={"osm_path": "label_path"})
    osm_sub = osm[osm["patch_path"].isin(sub["patch_path"])].copy()
    print(f"[info] {len(osm_sub)} matching mask entries")

    # write subset indexes to a temp dir under packaging/
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_idx = OUT_DIR / "_index.parquet"
    sub_osm = OUT_DIR / "_osm_index.parquet"
    sub.to_parquet(sub_idx)
    osm_sub.to_parquet(sub_osm)

    out_name = args.out or f"linhe_{args.quarters[0]}_{args.quarters[-1]}.tar.gz"
    out_path = OUT_DIR / out_name
    print(f"[info] writing {out_path}")

    with tarfile.open(out_path, "w:gz") as tar:
        # subset indexes (placed at the same relative path as on local disk)
        tar.add(sub_idx, arcname="data/linhe_patches/_index.parquet")
        tar.add(sub_osm, arcname="data/linhe_patches/_osm_index.parquet")
        # patches
        for path in sub["patch_path"]:
            full = ROOT / path
            tar.add(full, arcname=path.replace("\\", "/"))
        # masks
        for path in osm_sub["label_path"]:
            full = ROOT / path
            tar.add(full, arcname=path.replace("\\", "/"))
        # provenance
        manifest = OUT_DIR / "_manifest.txt"
        manifest.write_text(
            f"quarters: {args.quarters}\n"
            f"scenes: {sub['scene_id'].nunique()}\n"
            f"patches: {len(sub)}\n",
            encoding="utf-8",
        )
        tar.add(manifest, arcname="data/linhe_patches/_colab_manifest.txt")

    sz = out_path.stat().st_size
    print(f"[ok] tar.gz size: {sz / 1e6:.1f} MB")

    # SHA256
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    sha = h.hexdigest()
    sha_path = out_path.with_suffix(out_path.suffix + ".sha256")
    # Use binary write so Python on Windows does not turn \n into \r\n —
    # `sha256sum -c` on Linux treats the trailing \r as part of the filename.
    sha_path.write_bytes(f"{sha}  {out_path.name}\n".encode("utf-8"))
    print(f"[ok] sha256: {sha}")
    print(f"[ok] checksum file: {sha_path}")
    print()
    print(f"Upload to Drive, then in Colab:")
    print(f"  !cp /content/drive/MyDrive/{out_path.name} /content/")
    print(f"  !sha256sum -c /content/drive/MyDrive/{sha_path.name}")
    print(f"  !cd / && tar xzf /content/{out_path.name}")


if __name__ == "__main__":
    main()
