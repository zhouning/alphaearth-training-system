from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch


LULC_CLASS_NAMES = ["background", "built", "crops", "trees", "water", "rangeland_bare"]


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be used for LULC segmentation inference."""


def prepare_rgb_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an RGB image array into a normalized [1, 3, H, W] float tensor."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape [H, W, 3].")

    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
        if array.max(initial=0.0) > 1.0:
            array = array / 255.0

    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def summarize_mask(
    mask: np.ndarray,
    class_names: list[str] = LULC_CLASS_NAMES,
) -> dict[str, dict[str, int | float]]:
    """Return per-class pixel counts and area fractions for a class-id mask."""
    mask_array = np.asarray(mask)
    total_pixels = int(mask_array.size)
    denominator = max(total_pixels, 1)

    counts: dict[str, int] = {}
    fractions: dict[str, float] = {}
    for class_id, class_name in enumerate(class_names):
        count = int(np.count_nonzero(mask_array == class_id))
        counts[class_name] = count
        fractions[class_name] = count / denominator

    return {
        "class_pixel_counts": counts,
        "class_area_fraction": fractions,
    }


def summarize_logits_confidence(logits: torch.Tensor) -> dict[str, float | str]:
    """Summarize max-softmax confidence for a segmentation logits tensor."""
    probabilities = torch.softmax(logits, dim=1)
    max_probabilities = probabilities.max(dim=1).values.detach().cpu().numpy()
    return {
        "mean_max_probability": float(np.mean(max_probabilities)),
        "min_max_probability": float(np.min(max_probabilities)),
        "max_max_probability": float(np.max(max_probabilities)),
        "note": "softmax confidence is not calibrated",
    }


def compute_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    class_names: list[str] = LULC_CLASS_NAMES,
    ignore_index: int | None = None,
) -> dict:
    """Compute pixel accuracy and per-class IoU for two class-id masks."""
    pred_array = np.asarray(prediction)
    target_array = np.asarray(target)
    if pred_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction and target masks must have the same shape, got "
            f"{pred_array.shape} and {target_array.shape}."
        )

    valid = np.ones(target_array.shape, dtype=bool)
    if ignore_index is not None:
        valid = target_array != ignore_index

    valid_count = int(np.count_nonzero(valid))
    correct = int(np.count_nonzero((pred_array == target_array) & valid))
    pixel_accuracy = correct / max(valid_count, 1)

    per_class_iou: dict[str, float | None] = {}
    valid_ious: list[float] = []
    for class_id, class_name in enumerate(class_names):
        pred_class = (pred_array == class_id) & valid
        target_class = (target_array == class_id) & valid
        intersection = int(np.count_nonzero(pred_class & target_class))
        union = int(np.count_nonzero(pred_class | target_class))
        if union == 0:
            per_class_iou[class_name] = None
            continue
        iou = intersection / union
        per_class_iou[class_name] = iou
        valid_ious.append(iou)

    return {
        "pixel_accuracy": pixel_accuracy,
        "mIoU": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "per_class_iou": per_class_iou,
        "valid_pixel_count": valid_count,
        "ignore_index": ignore_index,
    }


def infer_adapter_kind(adapter_state: dict) -> str:
    """Infer which input adapter class matches a saved adapter state dict."""
    if any(key.startswith("channel_proj.") or key == "residual_scale" for key in adapter_state):
        return "geo_adapter"
    return "zero_pad"


def load_benchmark_checkpoint(path: str | Path) -> dict:
    """Load and validate a benchmark-style segmentation checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise CheckpointCompatibilityError(
            "Expected a benchmark-style segmentation checkpoint dictionary."
        )

    required_keys = {"adapter", "head"}
    if not required_keys.issubset(checkpoint.keys()):
        raise CheckpointCompatibilityError(
            "Expected a benchmark-style segmentation checkpoint with adapter and head weights."
        )

    head_state = checkpoint.get("head")
    if not isinstance(head_state, dict) or not any(key.startswith("proj.") for key in head_state):
        raise CheckpointCompatibilityError(
            "Expected a benchmark-style segmentation checkpoint with SegmentationHead weights."
        )

    proj_weight = head_state.get("proj.weight")
    if proj_weight is None or int(proj_weight.shape[0]) != len(LULC_CLASS_NAMES):
        raise CheckpointCompatibilityError(
            f"Expected a benchmark-style segmentation checkpoint with {len(LULC_CLASS_NAMES)} LULC classes."
        )

    backbone_peft = checkpoint.get("backbone_peft") or {}
    unsupported_peft_keys = [
        key for key in backbone_peft
        if "lora_" in key.lower()
    ]
    if unsupported_peft_keys:
        raise CheckpointCompatibilityError(
            "LoRA segmentation checkpoints are not supported by this inference API yet. "
            "Use a linear-probe, GeoAdapter, or Houlsby benchmark-style segmentation checkpoint."
        )

    return checkpoint


class LULCInferenceService:
    """Semantic-segmentation inference service for LULC image patches."""

    def __init__(
        self,
        predictor: Callable[[torch.Tensor], torch.Tensor] | None = None,
        model_id: str | None = None,
        device: str | None = None,
    ):
        if predictor is None:
            raise CheckpointCompatibilityError(
                "No LULC segmentation predictor is configured. Provide a compatible "
                "benchmark-style segmentation checkpoint."
            )
        self._predictor = predictor
        self.model_id = model_id or "lulc-segmentation"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        prithvi_checkpoint_path: str | Path | None = None,
        model_id: str | None = None,
        device: str | None = None,
    ) -> "LULCInferenceService":
        """Build a Prithvi + adapter + segmentation-head predictor from a checkpoint."""
        from geoadapter.adapters.geo_adapter import GeoAdapter
        from geoadapter.adapters.houlsby import inject_houlsby_adapters
        from geoadapter.adapters.zero_pad import ZeroPadAdapter
        from geoadapter.models.heads import SegmentationHead
        from geoadapter.models.prithvi import PrithviBackbone

        runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = load_benchmark_checkpoint(checkpoint_path)

        backbone = PrithviBackbone(
            pretrained=prithvi_checkpoint_path is not None,
            checkpoint_path=str(prithvi_checkpoint_path) if prithvi_checkpoint_path else None,
            in_chans=6,
        )
        for block in backbone.blocks:
            inject_houlsby_adapters(block)

        adapter_kind = infer_adapter_kind(checkpoint["adapter"])
        if adapter_kind == "geo_adapter":
            adapter = GeoAdapter(in_channels=3, out_channels=6)
        else:
            adapter = ZeroPadAdapter(in_channels=3, out_channels=6)
        head = SegmentationHead(
            in_dim=768,
            num_classes=len(LULC_CLASS_NAMES),
            patch_size=16,
        )

        adapter.load_state_dict(checkpoint["adapter"], strict=False)
        head.load_state_dict(checkpoint["head"])
        if checkpoint.get("backbone_peft"):
            backbone.load_state_dict(checkpoint["backbone_peft"], strict=False)

        backbone.to(runtime_device).eval()
        adapter.to(runtime_device).eval()
        head.to(runtime_device).eval()

        @torch.no_grad()
        def predictor(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.to(runtime_device)
            features, spatial_dims = backbone(adapter(tensor), return_spatial=True)
            return head(features, spatial_dims)

        return cls(
            predictor=predictor,
            model_id=model_id or Path(checkpoint_path).stem,
            device=runtime_device,
        )

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor) -> dict:
        logits = self._predictor(tensor)
        if logits.ndim != 4 or logits.shape[0] != 1:
            raise ValueError("Expected predictor logits with shape [1, C, H, W].")

        mask = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int64)
        summary = summarize_mask(mask)
        return {
            "task": "lulc_segmentation",
            "mode": "local_model",
            "classes": LULC_CLASS_NAMES,
            "model_id": self.model_id,
            "device": self.device,
            "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "mask": mask.tolist(),
            "confidence_summary": summarize_logits_confidence(logits),
            **summary,
        }

    def predict_image(self, image: np.ndarray) -> dict:
        return self.predict_tensor(prepare_rgb_tensor(image))
