"""Quarterly change detection for Linhe RGB patches (Task 1.c).

Reads paired patches from _pairs.parquet (or generates pairs on-the-fly for
specified quarters), computes pixel-level change maps using RGB difference and
PCA-based anomaly detection (RX-like), and outputs visualization PNGs + a
summary CSV with per-patch anomaly scores.

Methods:
  1. RGB L2 difference (Euclidean distance in RGB space, normalized)
  2. PCA-RX: project 6-band stack [R1,G1,B1,R2,G2,B2] into PCA space,
     compute Mahalanobis distance on residual components → anomaly score

Outputs:
  results/linhe_change/{qa}_vs_{qb}/
    change_rgb_diff_{row}_{col}.png   — RGB diff heatmap
    change_pca_rx_{row}_{col}.png     — PCA-RX anomaly heatmap
    pair_visual_{row}_{col}.png       — side-by-side comparison (A | B | diff)
  results/linhe_change/change_scores.csv — per-pair anomaly statistics

Usage:
  python scripts/linhe_change_detect.py --qa 2019Q4 --qb 2023Q1
  python scripts/linhe_change_detect.py --qa 2025Q1 --qb 2025Q4 --top-k 50
  python scripts/linhe_change_detect.py --qa 2025Q1 --qb 2025Q4 --rebuild-pairs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "linhe_patches" / "_index.parquet"
PAIRS_PATH = ROOT / "data" / "linhe_patches" / "_pairs.parquet"
OUT_ROOT = ROOT / "results" / "linhe_change"


def load_patch(path: str | Path) -> np.ndarray:
    """Load a patch .npz → (3, 128, 128) uint8 array."""
    full = ROOT / path if not Path(path).is_absolute() else Path(path)
    return np.load(full)["rgb"]


def rgb_l2_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance in RGB space, normalized to [0, 1]."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    dist = np.sqrt((diff ** 2).sum(axis=0))  # (H, W)
    max_dist = np.sqrt(3.0) * 255.0
    return dist / max_dist


def pca_rx_anomaly(a: np.ndarray, b: np.ndarray, n_components: int = 4) -> np.ndarray:
    """PCA-RX anomaly detection on stacked 6-band [RGB_a, RGB_b].

    Projects the 6-band pixel stack into PCA space, keeps minor components
    (residuals), and computes squared Mahalanobis distance as anomaly score.
    """
    h, w = a.shape[1], a.shape[2]
    stack = np.concatenate([a, b], axis=0).astype(np.float32)  # (6, H, W)
    pixels = stack.reshape(6, -1).T  # (N, 6)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(pixels)  # (N, n_components)

    cov = np.cov(transformed, rowvar=False)
    cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(n_components))
    mean = transformed.mean(axis=0)
    diff = transformed - mean
    mahal = np.sum(diff @ cov_inv * diff, axis=1)  # (N,)

    score_map = mahal.reshape(h, w)
    p99 = np.percentile(score_map, 99)
    if p99 > 0:
        score_map = np.clip(score_map / p99, 0, 1)
    return score_map


def get_pairs_for_quarters(qa: str, qb: str, iou_threshold: float = 0.8) -> pd.DataFrame:
    """Get or build patch pairs for two quarters."""
    if PAIRS_PATH.exists():
        pairs = pd.read_parquet(PAIRS_PATH)
        subset = pairs[
            ((pairs["quarter_a"] == qa) & (pairs["quarter_b"] == qb)) |
            ((pairs["quarter_a"] == qb) & (pairs["quarter_b"] == qa))
        ].copy()
        if len(subset) > 0:
            mask_swap = subset["quarter_a"] == qb
            if mask_swap.any():
                for col in ["patch_path", "scene_id", "quarter", "date"]:
                    a_col, b_col = f"{col}_a", f"{col}_b"
                    subset.loc[mask_swap, [a_col, b_col]] = (
                        subset.loc[mask_swap, [b_col, a_col]].values
                    )
            return subset.reset_index(drop=True)

    # Build pairs on-the-fly from index
    idx = pd.read_parquet(INDEX_PATH)
    ga = idx[idx["quarter"] == qa].copy()
    gb = idx[idx["quarter"] == qb].copy()
    if ga.empty or gb.empty:
        print(f"No patches for {qa} or {qb} in index.")
        return pd.DataFrame()

    from shapely.geometry import box as sbox
    from shapely.strtree import STRtree

    gb_geoms = [sbox(r.minx, r.miny, r.maxx, r.maxy) for _, r in gb.iterrows()]
    tree = STRtree(gb_geoms)

    rows = []
    for _, ra in ga.iterrows():
        ga_box = sbox(ra.minx, ra.miny, ra.maxx, ra.maxy)
        candidates = tree.query(ga_box)
        for ci in candidates:
            rb = gb.iloc[ci]
            gb_box = gb_geoms[ci]
            inter = ga_box.intersection(gb_box).area
            union = ga_box.union(gb_box).area
            iou = inter / union if union > 0 else 0
            if iou >= iou_threshold:
                rows.append({
                    "patch_path_a": ra.patch_path,
                    "patch_path_b": rb.patch_path,
                    "scene_id_a": ra.scene_id,
                    "scene_id_b": rb.scene_id,
                    "quarter_a": qa,
                    "quarter_b": qb,
                    "iou": iou,
                })
    return pd.DataFrame(rows)


def save_visualizations(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    rgb_diff: np.ndarray,
    pca_score: np.ndarray,
    out_dir: Path,
    row: int,
    col: int,
) -> None:
    """Save comparison and heatmap PNGs using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    tag = f"{row:05d}_{col:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = patch_a.transpose(1, 2, 0)  # CHW → HWC
    img_b = patch_b.transpose(1, 2, 0)

    # Side-by-side: A | B | RGB diff
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_a)
    axes[0].set_title("Before")
    axes[0].axis("off")
    axes[1].imshow(img_b)
    axes[1].set_title("After")
    axes[1].axis("off")
    axes[2].imshow(rgb_diff, cmap="hot", norm=Normalize(0, 1))
    axes[2].set_title("RGB Diff")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"pair_visual_{tag}.png", dpi=100, bbox_inches="tight")
    plt.close()

    # PCA-RX heatmap
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(pca_score, cmap="inferno", norm=Normalize(0, 1))
    ax.set_title("PCA-RX Anomaly")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.savefig(out_dir / f"change_pca_rx_{tag}.png", dpi=100, bbox_inches="tight")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Linhe quarterly change detection (1.c)")
    p.add_argument("--qa", required=True, help="Earlier quarter (e.g. 2019Q4)")
    p.add_argument("--qb", required=True, help="Later quarter (e.g. 2023Q1)")
    p.add_argument("--top-k", type=int, default=30,
                   help="Visualize top-K most changed pairs")
    p.add_argument("--iou-threshold", type=float, default=0.8,
                   help="Min IoU for pairing patches across quarters")
    p.add_argument("--rebuild-pairs", action="store_true",
                   help="Force rebuild pairs from index instead of using _pairs.parquet")
    args = p.parse_args()

    print(f"Change detection: {args.qa} -> {args.qb}")

    if args.rebuild_pairs or not PAIRS_PATH.exists():
        pairs = get_pairs_for_quarters(args.qa, args.qb, args.iou_threshold)
    else:
        pairs = get_pairs_for_quarters(args.qa, args.qb, args.iou_threshold)

    if pairs.empty:
        print("No pairs found. Run linhe_pair_patches.py first or use --rebuild-pairs.")
        return

    print(f"Found {len(pairs)} pairs for {args.qa} vs {args.qb}")

    out_dir = OUT_ROOT / f"{args.qa}_vs_{args.qb}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    for i, row in pairs.iterrows():
        patch_a = load_patch(row["patch_path_a"])
        patch_b = load_patch(row["patch_path_b"])

        diff_map = rgb_l2_diff(patch_a, patch_b)
        pca_map = pca_rx_anomaly(patch_a, patch_b)

        mean_diff = float(diff_map.mean())
        mean_pca = float(pca_map.mean())
        max_pca = float(pca_map.max())

        scores.append({
            "patch_path_a": row["patch_path_a"],
            "patch_path_b": row["patch_path_b"],
            "quarter_a": args.qa,
            "quarter_b": args.qb,
            "mean_rgb_diff": mean_diff,
            "mean_pca_score": mean_pca,
            "max_pca_score": max_pca,
        })

        if i < 5 or (i % 200 == 0):
            print(f"  [{i+1}/{len(pairs)}] diff={mean_diff:.4f} pca={mean_pca:.4f}")

    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values("mean_pca_score", ascending=False).reset_index(drop=True)

    csv_path = OUT_ROOT / "change_scores.csv"
    scores_df.to_csv(csv_path, index=False)
    print(f"\nScores saved: {csv_path} ({len(scores_df)} pairs)")

    # Visualize top-K
    top_k = min(args.top_k, len(scores_df))
    print(f"Generating visualizations for top-{top_k} changed patches...")
    for i in range(top_k):
        rec = scores_df.iloc[i]
        pa = load_patch(rec["patch_path_a"])
        pb = load_patch(rec["patch_path_b"])
        diff_map = rgb_l2_diff(pa, pb)
        pca_map = pca_rx_anomaly(pa, pb)

        # Extract row/col from patch path
        fname = Path(rec["patch_path_a"]).stem  # p_00128_01152
        parts = fname.split("_")
        r, c = int(parts[1]), int(parts[2])
        save_visualizations(pa, pb, diff_map, pca_map, out_dir, r, c)

    print(f"Done. Visualizations in: {out_dir}")
    print(f"\nTop-5 most changed patches:")
    print(scores_df[["patch_path_a", "mean_rgb_diff", "mean_pca_score"]].head())

    # Spatial summary: GeoJSON with change scores at patch centroids
    generate_spatial_summary(scores_df, args.qa, args.qb)


def generate_spatial_summary(scores_df: pd.DataFrame, qa: str, qb: str) -> None:
    """Join scores with index bbox, emit GeoJSON for map overlay."""
    import json
    from pyproj import Transformer

    idx = pd.read_parquet(INDEX_PATH)
    idx_lookup = idx.set_index("patch_path")[["minx", "miny", "maxx", "maxy"]].to_dict("index")

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    features = []
    for _, row in scores_df.iterrows():
        bbox = idx_lookup.get(row["patch_path_a"])
        if bbox is None:
            continue
        cx = (bbox["minx"] + bbox["maxx"]) / 2
        cy = (bbox["miny"] + bbox["maxy"]) / 2
        lon, lat = transformer.transform(cx, cy)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            "properties": {
                "mean_rgb_diff": round(row["mean_rgb_diff"], 4),
                "mean_pca_score": round(row["mean_pca_score"], 4),
                "patch_a": row["patch_path_a"],
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    out_path = OUT_ROOT / f"change_heatmap_{qa}_vs_{qb}.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    print(f"\nSpatial summary: {out_path} ({len(features)} points)")

    # Also generate a static matplotlib spatial plot
    if features:
        import matplotlib.pyplot as plt

        lons = [f["geometry"]["coordinates"][0] for f in features]
        lats = [f["geometry"]["coordinates"][1] for f in features]
        scores = [f["properties"]["mean_pca_score"] for f in features]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sc = ax.scatter(lons, lats, c=scores, cmap="hot_r", s=4, alpha=0.7,
                        vmin=0, vmax=max(scores) * 0.8)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Change Intensity: {qa} -> {qb}")
        plt.colorbar(sc, ax=ax, label="PCA-RX Score")
        plt.tight_layout()
        fig_path = OUT_ROOT / f"change_spatial_{qa}_vs_{qb}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Spatial plot: {fig_path}")


if __name__ == "__main__":
    main()
