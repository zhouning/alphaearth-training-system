from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image


PALETTE = np.array(
    [
        [32, 32, 32],     # background
        [204, 83, 75],    # built
        [224, 185, 76],   # crops
        [45, 156, 89],    # trees
        [64, 137, 201],   # water
        [151, 176, 96],   # rangeland_bare
    ],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    clipped = np.clip(mask, 0, len(PALETTE) - 1)
    return PALETTE[clipped]


def make_triptych(rgb: np.ndarray, label: np.ndarray, pred: np.ndarray) -> Image.Image:
    rgb_img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    label_img = Image.fromarray(colorize(label), mode="RGB")
    pred_img = Image.fromarray(colorize(pred), mode="RGB")
    canvas = Image.new("RGB", (rgb_img.width * 3, rgb_img.height))
    canvas.paste(rgb_img, (0, 0))
    canvas.paste(label_img, (rgb_img.width, 0))
    canvas.paste(pred_img, (rgb_img.width * 2, 0))
    return canvas


def ensure_hwc_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB array in CHW or HWC layout, got {rgb.shape}")
    return rgb.astype(np.uint8)


def compute_patch_metrics(prediction: np.ndarray, label: np.ndarray, n_classes: int) -> dict:
    prediction = np.asarray(prediction, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    if prediction.shape != label.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match label shape {label.shape}")

    total = max(int(label.size), 1)
    pixel_accuracy = float(np.count_nonzero(prediction == label) / total)
    per_class_iou: dict[str, float | None] = {}
    ious: list[float] = []
    for class_id in range(n_classes):
        pred_class = prediction == class_id
        label_class = label == class_id
        union = int(np.count_nonzero(pred_class | label_class))
        if union == 0:
            per_class_iou[f"class_{class_id}"] = None
            continue
        intersection = int(np.count_nonzero(pred_class & label_class))
        iou = float(intersection / union)
        per_class_iou[f"class_{class_id}"] = iou
        ious.append(iou)
    return {
        "pixel_accuracy": pixel_accuracy,
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "per_class_iou": per_class_iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8087/api/ae/inference/lulc")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--patch-row", type=int, default=0)
    parser.add_argument("--output-dir", default="results/lulc_api_smoke")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_parquet(repo / "data" / "linhe_patches" / "_lulc_index.parquet")
    rows = index[index["year"] == 2022].reset_index(drop=True)
    row = rows.iloc[args.patch_row]

    patch = np.load(repo / row["patch_path"])
    label = np.load(repo / row["lulc_path"])["mask"]
    rgb = ensure_hwc_rgb(patch["rgb"])

    png = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(png, format="PNG")
    png.seek(0)

    session = requests.Session()
    session.trust_env = False
    data = {}
    if args.checkpoint_path:
        data["checkpoint_path"] = args.checkpoint_path
    if args.model_id:
        data["model_id"] = args.model_id

    response = session.post(
        args.api_url,
        data=data,
        files={"file": ("linhe_patch.png", png, "image/png")},
        timeout=args.timeout,
    )
    response.raise_for_status()
    result = response.json()

    pred = np.asarray(result["mask"], dtype=np.int64)
    patch_metrics = compute_patch_metrics(pred, label, n_classes=len(result.get("classes", [])) or 6)
    triptych = make_triptych(rgb, label, pred)
    triptych_path = output_dir / "linhe_patch_rgb_label_prediction.png"
    result_path = output_dir / "linhe_patch_prediction.json"
    triptych.save(triptych_path)
    result_path.write_text(json.dumps({
            "source_patch": row["patch_path"],
            "source_label": row["lulc_path"],
            "patch_metrics": patch_metrics,
            "api_result": result,
        }, indent=2), encoding="utf-8")

    print(json.dumps({
        "triptych": str(triptych_path),
        "result": str(result_path),
        "patch_metrics": patch_metrics,
        "class_pixel_counts": result["class_pixel_counts"],
        "class_area_fraction": result["class_area_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
