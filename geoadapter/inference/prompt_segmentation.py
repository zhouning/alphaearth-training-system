from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rasterio
import torch
from PIL import Image

from geoadapter.bench.run_geovlm_prompt_segmentation import (
    PROMPT_METHOD,
    build_model,
    checkpoint_metadata,
    load_config,
    validate_checkpoint_metadata,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "bench"
    / "configs"
    / "geovlm_prompt_segmentation.yaml"
)
VALIDATED_SEMANTIC_SCOPE = ["building", "road", "water"]


@dataclass(frozen=True)
class PromptImage:
    tensor: torch.Tensor
    rgb: np.ndarray
    crs: object | None
    transform: object | None
    source_path: Path


def load_prompt_image(path: str | Path) -> PromptImage:
    path = Path(path)
    suffix = path.suffix.lower()
    supported = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    if suffix not in supported:
        raise ValueError(f"unsupported image extension: {path.suffix}")
    if not path.is_file():
        raise FileNotFoundError(f"input image not found: {path}")
    if suffix in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            if source.count < 3:
                raise ValueError("GeoTIFF must contain at least three RGB bands")
            rgb = np.moveaxis(source.read([1, 2, 3]), 0, -1)
            crs = source.crs
            transform = source.transform
    elif suffix in {".png", ".jpg", ".jpeg"}:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        crs = None
        transform = None
    if not np.isfinite(rgb).all():
        raise ValueError("input image contains non-finite values")
    scaled = rgb.astype(np.float32)
    if scaled.size == 0 or scaled.min() < 0 or scaled.max() > 255:
        raise ValueError("MVP RGB values must be in 0..255")
    scaled /= 255.0
    tensor = torch.from_numpy(np.moveaxis(scaled, -1, 0).copy()).contiguous()
    display_rgb = np.rint(scaled * 255.0).astype(np.uint8)
    return PromptImage(tensor, display_rgb, crs, transform, path)


def _config_copy(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    return load_config(config)


def load_prompt_checkpoint(
    checkpoint_path: str | Path,
    config: Mapping[str, Any] | str | Path = DEFAULT_CONFIG_PATH,
    *,
    device: str = "cpu",
    local_files_only: bool = False,
    model_builder=build_model,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"prompt checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("metadata"), dict):
        raise ValueError("prompt checkpoint is missing metadata")
    metadata = dict(state["metadata"])
    if metadata.get("method") != PROMPT_METHOD:
        raise ValueError("offline prompt inference requires a prompt checkpoint")

    resolved_config = _config_copy(config)
    resolved_config["text_encoder"]["local_files_only"] = bool(local_files_only)
    expected = checkpoint_metadata(
        resolved_config,
        PROMPT_METHOD,
        int(metadata.get("seed", -1)),
    )
    validate_checkpoint_metadata(metadata, expected)
    model = model_builder(resolved_config, PROMPT_METHOD, device)
    trainable_state = state.get("trainable_model")
    if not isinstance(trainable_state, dict):
        raise ValueError("prompt checkpoint is missing trainable_model state")
    expected_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    actual_names = set(trainable_state)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            "prompt checkpoint trainable state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    result = model.load_state_dict(trainable_state, strict=False)
    if result.unexpected_keys:
        raise ValueError(
            f"unexpected prompt checkpoint parameters: {result.unexpected_keys}"
        )
    model.eval()
    return model, metadata


@torch.no_grad()
def run_prompt_inference(
    model,
    image: PromptImage,
    prompt: str,
    *,
    threshold: float = 0.5,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty English text")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    logits = model(image.tensor.unsqueeze(0).to(device), [prompt])
    if tuple(logits.shape) != (1, *image.tensor.shape[-2:]):
        raise ValueError(
            "prompt model must return logits with shape [1,H,W] matching the input"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("prompt model returned non-finite logits")
    probability = logits.sigmoid()[0].detach().cpu().numpy().astype(np.float32)
    mask = (probability >= threshold).astype(np.uint8)
    return probability, mask


def _preview_rgb(image: PromptImage, probability: np.ndarray, mask: np.ndarray):
    probability_rgb = np.repeat(
        np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    mask_rgb = np.repeat(mask[..., None] * 255, 3, axis=2).astype(np.uint8)
    separator = np.full((image.rgb.shape[0], 2, 3), 255, dtype=np.uint8)
    return np.concatenate(
        (image.rgb, separator, probability_rgb, separator, mask_rgb), axis=1
    )


def _write_outputs(
    image: PromptImage,
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    prompt: str,
    threshold: float,
    checkpoint: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image.source_path.stem
    probability_path = output_dir / f"{stem}_probability.npy"
    preview_path = output_dir / f"{stem}_preview.png"
    metadata_path = output_dir / f"{stem}_metadata.json"
    georeferenced = image.crs is not None and image.transform is not None
    if georeferenced:
        mask_path = output_dir / f"{stem}_mask.tif"
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            width=mask.shape[1],
            height=mask.shape[0],
            count=1,
            dtype="uint8",
            crs=image.crs,
            transform=image.transform,
            compress="deflate",
        ) as destination:
            destination.write(mask, 1)
    else:
        mask_path = output_dir / f"{stem}_mask.png"
        Image.fromarray(mask * 255).save(mask_path)
    np.save(probability_path, probability, allow_pickle=False)
    Image.fromarray(_preview_rgb(image, probability, mask)).save(preview_path)

    metadata = {
        "schema": "paper12.geovlm_prompt_inference.v1",
        "prompt": prompt,
        "validated_semantic_scope": list(VALIDATED_SEMANTIC_SCOPE),
        "method": checkpoint["method"],
        "seed": int(checkpoint["seed"]),
        "checkpoint_schema": checkpoint["schema"],
        "prithvi_sha256": checkpoint["prithvi_sha256"],
        "prompt_config_sha256": checkpoint["prompt_config_sha256"],
        "siglip_model_id": checkpoint["siglip_model_id"],
        "siglip_revision": checkpoint.get("siglip_revision"),
        "image_normalization": checkpoint["image_normalization"],
        "input_path": str(image.source_path),
        "input_height": int(mask.shape[0]),
        "input_width": int(mask.shape[1]),
        "threshold": float(threshold),
        "foreground_pixel_share": float(mask.mean()),
        "georeferenced": georeferenced,
        "crs": str(image.crs) if georeferenced else None,
        "transform": list(image.transform) if georeferenced else None,
        "output_paths": {
            "mask": str(mask_path),
            "probability": str(probability_path),
            "preview": str(preview_path),
            "metadata": str(metadata_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def predict_prompt_image(
    image_path: str | Path,
    prompt: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | str | Path = DEFAULT_CONFIG_PATH,
    threshold: float | None = None,
    device: str = "cpu",
    local_files_only: bool = False,
    model_builder=build_model,
) -> dict[str, Any]:
    resolved_config = _config_copy(config)
    selected_threshold = (
        float(resolved_config["evaluation"]["threshold"])
        if threshold is None
        else float(threshold)
    )
    image = load_prompt_image(image_path)
    model, metadata = load_prompt_checkpoint(
        checkpoint_path,
        resolved_config,
        device=device,
        local_files_only=local_files_only,
        model_builder=model_builder,
    )
    probability, mask = run_prompt_inference(
        model,
        image,
        prompt,
        threshold=selected_threshold,
        device=device,
    )
    return _write_outputs(
        image,
        probability,
        mask,
        prompt=prompt,
        threshold=selected_threshold,
        checkpoint=metadata,
        output_dir=output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline Paper12 GeoVLM prompt segmentation"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    predict_prompt_image(
        args.image,
        args.prompt,
        args.checkpoint,
        args.output_dir,
        config=args.config,
        threshold=args.threshold,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
