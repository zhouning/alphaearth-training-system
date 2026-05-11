"""Quality-check Linhe LULC patch coverage.

Reads:
  data/linhe_patches/_index.parquet        (RGB patches, ground truth set)
  data/linhe_patches/_lulc_index.parquet   (LULC masks pulled by linhe_pull_esri_lulc.py)
  results/linhe/esri_lulc/_manifest.json   (class name table)

Reports:
  - per-year class pixel share + minority-class flag
  - patches missing a LULC mask (per year)
  - per-scene coverage matrix (scene × year)
  - 2021→2022 building-area delta (preview for change-detection 1.c)

Outputs:
  results/linhe/esri_lulc/coverage_report.md
  results/linhe/esri_lulc/coverage_per_year.csv
  results/linhe/esri_lulc/coverage_per_scene.csv

Usage:
  python scripts/linhe_check_lulc_coverage.py
  python scripts/linhe_check_lulc_coverage.py --min-share 0.005  # flag classes below 0.5%
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATCH_INDEX = ROOT / "data" / "linhe_patches" / "_index.parquet"
LULC_INDEX = ROOT / "data" / "linhe_patches" / "_lulc_index.parquet"
MANIFEST = ROOT / "results" / "linhe" / "esri_lulc" / "_manifest.json"
OUT_DIR = ROOT / "results" / "linhe" / "esri_lulc"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not PATCH_INDEX.exists():
        raise SystemExit(f"[error] {PATCH_INDEX} not found — run linhe_build_patches.py first")
    if not LULC_INDEX.exists():
        raise SystemExit(f"[error] {LULC_INDEX} not found — run linhe_pull_esri_lulc.py first")
    if not MANIFEST.exists():
        raise SystemExit(f"[error] {MANIFEST} not found — run linhe_pull_esri_lulc.py first")
    patches = pd.read_parquet(PATCH_INDEX)
    lulc = pd.read_parquet(LULC_INDEX)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return patches, lulc, manifest


def pixel_counts(npz_path: Path, n_classes: int) -> np.ndarray:
    arr = np.load(npz_path)["mask"]
    counts = np.bincount(arr.ravel(), minlength=n_classes)
    return counts[:n_classes]


def per_year_class_share(lulc: pd.DataFrame, n_classes: int, class_names: dict[int, str]) -> pd.DataFrame:
    """Aggregate pixel counts across every patch mask, grouped by year."""
    by_year: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n_classes, dtype=np.int64))
    for _, r in lulc.iterrows():
        p = ROOT / r["lulc_path"]
        if not p.exists():
            continue
        try:
            by_year[int(r["year"])] += pixel_counts(p, n_classes)
        except Exception as e:
            print(f"[warn] {p}: {e}")

    rows = []
    for year, counts in sorted(by_year.items()):
        total = counts.sum()
        for cid in range(n_classes):
            rows.append({
                "year": year,
                "class_id": cid,
                "class_name": class_names.get(cid, class_names.get(str(cid), f"class_{cid}")),
                "pixels": int(counts[cid]),
                "share": float(counts[cid] / total) if total else 0.0,
            })
    return pd.DataFrame(rows)


def missing_patches(patches: pd.DataFrame, lulc: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Patches in _index.parquet that have no LULC mask for one or more years."""
    rows = []
    expected = set(patches["patch_path"])
    for year in years:
        got = set(lulc[lulc["year"] == year]["patch_path"])
        for p in sorted(expected - got):
            rows.append({"patch_path": p, "missing_year": year})
    return pd.DataFrame(rows)


def per_scene_coverage(patches: pd.DataFrame, lulc: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Scene × year: how many patches got a LULC mask vs how many should have."""
    expected = patches.groupby("scene_id").size().rename("n_patches")
    cov = (lulc.groupby(["scene_id", "year"]).size()
                .unstack("year", fill_value=0)
                .reindex(columns=years, fill_value=0))
    cov = cov.join(expected, how="outer").fillna(0).astype(int)
    for y in years:
        cov[f"{y}_pct"] = (cov[y] / cov["n_patches"]).round(3).where(cov["n_patches"] > 0, 0)
    return cov.sort_index()


def building_delta(lulc: pd.DataFrame, class_names: dict[int, str]) -> dict | None:
    """If both 2021 and 2022 are present, preview the 1.c building-change signal."""
    built_id = None
    for cid, name in class_names.items():
        cid_int = int(cid) if isinstance(cid, str) else cid
        if str(name).lower() in {"built", "built area", "built_area"}:
            built_id = cid_int
            break
    if built_id is None:
        return None
    years = sorted(lulc["year"].unique().tolist())
    if 2021 not in years or 2022 not in years:
        return None

    def built_pixels(year: int) -> int:
        total = 0
        sub = lulc[lulc["year"] == year]
        for _, r in sub.iterrows():
            p = ROOT / r["lulc_path"]
            if not p.exists():
                continue
            arr = np.load(p)["mask"]
            total += int((arr == built_id).sum())
        return total

    b21 = built_pixels(2021)
    b22 = built_pixels(2022)
    return {
        "built_class_id": built_id,
        "built_pixels_2021": b21,
        "built_pixels_2022": b22,
        "delta_pixels": b22 - b21,
        "delta_pct": (b22 - b21) / b21 if b21 else None,
    }


def write_report(per_year: pd.DataFrame, missing: pd.DataFrame, per_scene: pd.DataFrame,
                 delta: dict | None, min_share: float, manifest: dict) -> Path:
    lines = [
        "# Linhe ESRI LULC Coverage Report",
        "",
        f"- Built at: `{pd.Timestamp.now().isoformat(timespec='seconds')}`",
        f"- Years checked: `{sorted(per_year['year'].unique().tolist())}`",
        f"- Class scheme: `{'linhe_6' if manifest.get('n_classes') == 6 else 'esri_9'}` ({manifest.get('n_classes')} classes)",
        f"- Minority threshold (flag if share < {min_share})",
        "",
        "## Per-year class share",
        "",
        per_year.pivot(index="class_id", columns="year", values="share")
                .round(4).to_markdown(),
        "",
    ]
    minor = per_year[(per_year["share"] < min_share) & (per_year["pixels"] > 0)]
    if len(minor):
        lines += [
            f"### ⚠️  {len(minor)} minority (class, year) buckets below {min_share}",
            "",
            minor.to_markdown(index=False),
            "",
        ]

    lines += [
        "## Per-scene coverage",
        "",
        per_scene.to_markdown(),
        "",
    ]

    if delta is not None:
        lines += [
            "## 2021 → 2022 built-area delta (preview for 1.c change detection)",
            "",
            f"- Built class id: **{delta['built_class_id']}**",
            f"- 2021 built pixels: **{delta['built_pixels_2021']:,}**",
            f"- 2022 built pixels: **{delta['built_pixels_2022']:,}**",
            f"- Δ pixels: **{delta['delta_pixels']:+,}** "
            f"({delta['delta_pct']*100:+.2f}%)" if delta["delta_pct"] is not None else "",
            "",
        ]

    if len(missing):
        lines += [
            f"## Missing masks ({len(missing)} patch×year entries)",
            "",
            "First 20 rows:",
            "",
            missing.head(20).to_markdown(index=False),
        ]
    else:
        lines += ["## Missing masks", "", "_None — full coverage._"]

    out = OUT_DIR / "coverage_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-share", type=float, default=0.005,
                   help="flag classes whose annual share falls below this")
    args = p.parse_args()

    patches, lulc, manifest = load_inputs()
    n_classes = int(manifest.get("n_classes", 6))
    class_names = manifest.get("class_names", {}) or {}
    class_names = {int(k): v for k, v in class_names.items()}
    years = sorted(lulc["year"].unique().tolist())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] patches={len(patches)}, lulc_rows={len(lulc)}, years={years}, classes={n_classes}")

    per_year = per_year_class_share(lulc, n_classes, class_names)
    per_year.to_csv(OUT_DIR / "coverage_per_year.csv", index=False, encoding="utf-8-sig")
    print(f"[ok]   per-year shares → {OUT_DIR / 'coverage_per_year.csv'}")

    missing = missing_patches(patches, lulc, years)
    per_scene = per_scene_coverage(patches, lulc, years)
    per_scene.to_csv(OUT_DIR / "coverage_per_scene.csv", encoding="utf-8-sig")
    print(f"[ok]   per-scene coverage → {OUT_DIR / 'coverage_per_scene.csv'}")
    print(f"[info] missing patch×year entries: {len(missing)}")

    delta = building_delta(lulc, class_names)
    if delta:
        sign = "+" if delta["delta_pixels"] >= 0 else ""
        print(f"[info] 2021→2022 built Δ: {sign}{delta['delta_pixels']:,} pixels "
              f"({delta['delta_pct']*100:+.2f}%)" if delta["delta_pct"] is not None
              else f"[info] built pixels 2021={delta['built_pixels_2021']}, 2022={delta['built_pixels_2022']}")

    report = write_report(per_year, missing, per_scene, delta, args.min_share, manifest)
    print(f"[done] report → {report}")


if __name__ == "__main__":
    main()
