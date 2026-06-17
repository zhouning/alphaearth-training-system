import sys
from pathlib import Path

import numpy as np
import pytest
import torch


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_prepare_rgb_tensor_normalizes_uint8_image():
    from app.services.inference import prepare_rgb_tensor

    image = np.array(
        [
            [[0, 127, 255], [255, 0, 0]],
            [[10, 20, 30], [40, 50, 60]],
        ],
        dtype=np.uint8,
    )

    tensor = prepare_rgb_tensor(image)

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == torch.float32
    assert tensor[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert tensor[0, 2, 0, 0].item() == pytest.approx(1.0)


def test_summarize_mask_reports_counts_and_area_fractions_for_all_lulc_classes():
    from app.services.inference import LULC_CLASS_NAMES, summarize_mask

    mask = np.array(
        [
            [0, 1, 1],
            [3, 3, 5],
        ],
        dtype=np.int64,
    )

    summary = summarize_mask(mask)

    assert list(summary["class_pixel_counts"].keys()) == LULC_CLASS_NAMES
    assert summary["class_pixel_counts"] == {
        "background": 1,
        "built": 2,
        "crops": 0,
        "trees": 2,
        "water": 0,
        "rangeland_bare": 1,
    }
    assert summary["class_area_fraction"]["background"] == pytest.approx(1 / 6)
    assert summary["class_area_fraction"]["built"] == pytest.approx(2 / 6)
    assert summary["class_area_fraction"]["crops"] == pytest.approx(0.0)
    assert summary["class_area_fraction"]["trees"] == pytest.approx(2 / 6)
    assert summary["class_area_fraction"]["water"] == pytest.approx(0.0)
    assert summary["class_area_fraction"]["rangeland_bare"] == pytest.approx(1 / 6)


def test_load_benchmark_checkpoint_rejects_non_segmentation_checkpoint(tmp_path: Path):
    from app.services.inference import (
        CheckpointCompatibilityError,
        load_benchmark_checkpoint,
    )

    checkpoint_path = tmp_path / "embedding_checkpoint.pt"
    torch.save({"encoder": {"weight": torch.ones(1)}}, checkpoint_path)

    with pytest.raises(CheckpointCompatibilityError, match="benchmark-style segmentation"):
        load_benchmark_checkpoint(checkpoint_path)


def test_load_benchmark_checkpoint_rejects_non_lulc_class_count(tmp_path: Path):
    from app.services.inference import (
        CheckpointCompatibilityError,
        load_benchmark_checkpoint,
    )

    checkpoint_path = tmp_path / "loveda_7class_checkpoint.pt"
    torch.save(
        {
            "adapter": {},
            "head": {
                "proj.weight": torch.zeros((7, 768, 1, 1)),
                "proj.bias": torch.zeros(7),
            },
        },
        checkpoint_path,
    )

    with pytest.raises(CheckpointCompatibilityError, match="6 LULC classes"):
        load_benchmark_checkpoint(checkpoint_path)


def test_load_benchmark_checkpoint_rejects_unsupported_lora_peft_state(tmp_path: Path):
    from app.services.inference import (
        CheckpointCompatibilityError,
        load_benchmark_checkpoint,
    )

    checkpoint_path = tmp_path / "lora_segmentation_checkpoint.pt"
    torch.save(
        {
            "adapter": {},
            "head": {
                "proj.weight": torch.zeros((6, 768, 1, 1)),
                "proj.bias": torch.zeros(6),
            },
            "backbone_peft": {
                "blocks.0.self_attn.lora_A.weight": torch.zeros((8, 768)),
            },
        },
        checkpoint_path,
    )

    with pytest.raises(CheckpointCompatibilityError, match="LoRA"):
        load_benchmark_checkpoint(checkpoint_path)


def test_infer_adapter_kind_detects_geoadapter_checkpoint_state():
    from app.services.inference import infer_adapter_kind

    adapter_state = {
        "residual_scale": torch.zeros(1),
        "channel_proj.weight": torch.zeros((6, 3, 1, 1)),
        "channel_proj.bias": torch.zeros(6),
    }

    assert infer_adapter_kind(adapter_state) == "geo_adapter"
    assert infer_adapter_kind({}) == "zero_pad"


def test_lulc_service_predicts_mask_and_summary_with_injected_predictor():
    from app.services.inference import LULCInferenceService

    def predictor(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape == (1, 3, 2, 2)
        logits = torch.zeros((1, 6, 2, 2), dtype=torch.float32)
        logits[:, 2, :, :] = 1.0
        logits[:, 3, 0, 1] = 2.0
        return logits

    service = LULCInferenceService(
        predictor=predictor,
        model_id="unit-test-model",
        device="cpu",
    )

    result = service.predict_image(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result["task"] == "lulc_segmentation"
    assert result["classes"] == ["background", "built", "crops", "trees", "water", "rangeland_bare"]
    assert result["model_id"] == "unit-test-model"
    assert result["device"] == "cpu"
    assert result["mask_shape"] == [2, 2]
    assert result["mask"] == [[2, 3], [2, 2]]
    assert result["class_pixel_counts"]["crops"] == 3
    assert result["class_pixel_counts"]["trees"] == 1
    assert result["class_area_fraction"]["crops"] == pytest.approx(0.75)
    assert result["class_area_fraction"]["trees"] == pytest.approx(0.25)


def test_lulc_service_reports_confidence_summary_from_logits():
    from app.services.inference import LULCInferenceService

    def predictor(tensor: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((1, 6, 1, 2), dtype=torch.float32)
        logits[:, 2, :, :] = 5.0
        logits[:, 3, :, 1] = 7.0
        return logits

    service = LULCInferenceService(
        predictor=predictor,
        model_id="confidence-model",
        device="cpu",
    )

    result = service.predict_image(np.zeros((1, 2, 3), dtype=np.uint8))

    assert result["mode"] == "local_model"
    assert result["confidence_summary"]["mean_max_probability"] > 0.8
    assert result["confidence_summary"]["min_max_probability"] > 0.7
    assert result["confidence_summary"]["max_max_probability"] <= 1.0
    assert result["confidence_summary"]["note"] == "softmax confidence is not calibrated"


def test_compute_segmentation_metrics_reports_per_class_iou_and_accuracy():
    from app.services.inference import compute_segmentation_metrics

    pred = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.int64)
    target = np.array([[0, 1, 2], [2, 1, 2]], dtype=np.int64)

    metrics = compute_segmentation_metrics(pred, target, class_names=["background", "built", "crops"])

    assert metrics["pixel_accuracy"] == pytest.approx(4 / 6)
    assert metrics["per_class_iou"]["background"] == pytest.approx(1.0)
    assert metrics["per_class_iou"]["built"] == pytest.approx(1 / 3)
    assert metrics["per_class_iou"]["crops"] == pytest.approx(0.5)
    assert metrics["mIoU"] == pytest.approx((1.0 + 1 / 3 + 0.5) / 3)
