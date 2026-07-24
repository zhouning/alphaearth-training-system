from __future__ import annotations

import hashlib
import importlib.metadata
import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from geoadapter.data.prompt_segmentation import (
    PROMPT_TARGET_CLASS_IDS,
    PromptConfig,
    load_prompt_config,
    multiclass_to_binary,
    normalize_landcoverai_image,
    sample_prompt_batch,
    validate_landcoverai_mask,
)
from geoadapter.engine.prompt_segmentation import (
    PromptSegmentationLoss,
    PromptSegmentationTrainer,
)
from geoadapter.models.prompt_segmentation import (
    PromptSegmentationModel,
    ThreeHeadSegmentationBaseline,
)
from geoadapter.bench.geovlm_prompt_summary import build_summary, binary_metrics


PROMPT_METHOD = "siglip_film_dense_similarity_houlsby"
BASELINE_METHOD = "no_text_three_binary_heads_houlsby"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["method"]), int(row["seed"])) for row in rows}


def estimate_positive_weights(
    masks: Iterable[torch.Tensor], *, clip: tuple[float, float] = (1.0, 20.0)
) -> dict[str, float]:
    low, high = (float(value) for value in clip)
    if low <= 0 or high < low:
        raise ValueError("positive weight clip must satisfy 0 < low <= high")
    total = 0
    positive = {name: 0 for name in PROMPT_TARGET_CLASS_IDS}
    for mask in masks:
        validate_landcoverai_mask(mask)
        total += int(mask.numel())
        for name, class_id in PROMPT_TARGET_CLASS_IDS.items():
            positive[name] += int(mask.eq(class_id).sum())
    if total == 0:
        raise ValueError("at least one non-empty LandCoverAI mask is required")
    return {
        name: min(high, max(low, (total - count) / count if count else high))
        for name, count in positive.items()
    }


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("torch", "torchvision", "torchgeo", "transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def checkpoint_metadata(config: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    prithvi_path = Path(config["prithvi"]["checkpoint"])
    prompt_path = Path(config["experiment"]["prompt_config"])
    return {
        "schema": "paper12.geovlm_prompt_checkpoint.v2",
        "training_contract": config["experiment"]["training_contract"],
        "target_pool_policy": "supported_target_present_only",
        "empty_target_cap": float(config["experiment"]["empty_target_cap"]),
        "probe_positives_per_class": int(
            config["experiment"]["probe_positives_per_class"]
        ),
        "best_checkpoint_policy": "finite_nonconstant_prompt_change_loss_v1",
        "method": method,
        "seed": int(seed),
        "prithvi_sha256": sha256_file(prithvi_path),
        "position_policy": "mean_temporal_3x14x14_then_bilinear",
        "image_normalization": "rgb_float32_divide_255",
        "siglip_model_id": config["text_encoder"]["model_id"],
        "siglip_revision": config["text_encoder"].get("revision"),
        "prompt_config_sha256": sha256_file(prompt_path),
        "class_mapping": dict(PROMPT_TARGET_CLASS_IDS),
        "condition_dim": int(config["model"]["condition_dim"]),
        "decoder_dim": int(config["model"]["decoder_dim"]),
        "threshold": float(config["evaluation"]["threshold"]),
        "dependency_versions": dependency_versions(),
    }


class LandCoverAIPromptView(Dataset):
    """Normalize and validate LandCoverAI samples without changing legacy loaders."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        image = normalize_landcoverai_image(image)
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("normalized LandCoverAI image must be finite and in 0..1")
        mask = mask.long()
        validate_landcoverai_mask(mask)
        return image, mask


class InputAdaptedModel(nn.Module):
    def __init__(self, adapter: nn.Module, model: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.model = model

    def forward(self, images, conditions):
        return self.model(self.adapter(images), conditions)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_datasets(config: dict[str, Any]):
    from geoadapter.data.datasets import load_landcoverai

    root = config["experiment"]["dataset_root"]
    if config["experiment"]["dataset"] != "landcoverai":
        raise ValueError("GeoVLM runner supports only the LandCoverAI dataset")
    train = load_landcoverai(root=root, split="train")
    validation = load_landcoverai(root=root, split="val")
    return LandCoverAIPromptView(train), LandCoverAIPromptView(validation)


def build_text_encoder(config: dict[str, Any]):
    from geoadapter.models.text_encoder import SiglipTextEncoder

    text_encoder = config["text_encoder"]
    return SiglipTextEncoder(
        text_encoder["model_id"],
        revision=text_encoder.get("revision"),
        cache_dir=text_encoder.get("cache_dir"),
        local_files_only=bool(text_encoder.get("local_files_only", False)),
    )


def build_model(config: dict[str, Any], method: str, device: str = "cpu"):
    checkpoint = Path(config["prithvi"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Prithvi checkpoint not found: {checkpoint}")
    from geoadapter.adapters.houlsby import inject_houlsby_adapters
    from geoadapter.adapters.zero_pad import ZeroPadAdapter
    from geoadapter.models.prithvi import PrithviBackbone

    backbone = PrithviBackbone(
        checkpoint_path=str(checkpoint),
        in_chans=int(config["prithvi"]["input_channels"]),
        patch_size=int(config["prithvi"]["patch_size"]),
        use_checkpoint_position_embeddings=bool(
            config["prithvi"]["use_checkpoint_position_embeddings"]
        ),
    )
    for block in backbone.blocks:
        inject_houlsby_adapters(
            block, bottleneck_dim=int(config["peft"]["bottleneck_dim"])
        )
    if method == PROMPT_METHOD:
        text_encoder = build_text_encoder(config)
        model = PromptSegmentationModel(
            backbone,
            text_encoder,
            visual_dim=backbone.embed_dim,
            text_dim=text_encoder.output_dim,
            condition_dim=int(config["model"]["condition_dim"]),
            decoder_dim=int(config["model"]["decoder_dim"]),
            patch_size=int(config["prithvi"]["patch_size"]),
        )
    elif method == BASELINE_METHOD:
        model = ThreeHeadSegmentationBaseline(
            backbone,
            visual_dim=backbone.embed_dim,
            decoder_dim=int(config["model"]["decoder_dim"]),
            patch_size=int(config["prithvi"]["patch_size"]),
        )
    else:
        raise ValueError(f"unsupported GeoVLM method: {method}")
    wrapped = InputAdaptedModel(ZeroPadAdapter(3, 6), model).to(device)
    trainable, frozen = _parameter_counts(wrapped)
    if trainable <= 0 or frozen <= 0:
        raise RuntimeError(
            "GeoVLM model must contain both trainable and frozen parameters"
        )
    print(
        f"[geovlm] method={method} trainable_params={trainable} "
        f"frozen_params={frozen}"
    )
    return wrapped


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return trainable, frozen


def build_trainer(model, config: dict[str, Any], device: str):
    loss = PromptSegmentationLoss(
        bce_weight=float(config["training"]["bce_weight"]),
        dice_weight=float(config["training"]["dice_weight"]),
    )
    return PromptSegmentationTrainer(
        model,
        lr=float(config["training"]["lr"]),
        lr_peft=float(
            config["training"].get("lr_peft", config["training"]["lr"])
        ),
        epochs=int(config["experiment"]["epochs"]),
        device=device,
        loss=loss,
    )


def _positive_weights_for_batch(class_names, weights, device):
    return torch.tensor([weights[name] for name in class_names], dtype=torch.float32, device=device)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def validate_checkpoint_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    fields = (
        "schema",
        "training_contract",
        "target_pool_policy",
        "empty_target_cap",
        "probe_positives_per_class",
        "best_checkpoint_policy",
        "method",
        "seed",
        "prithvi_sha256",
        "position_policy",
        "image_normalization",
        "siglip_model_id",
        "siglip_revision",
        "prompt_config_sha256",
        "class_mapping",
        "condition_dim",
        "decoder_dim",
        "threshold",
    )
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ValueError(f"checkpoint metadata mismatch: {field}")


def seed42_smoke_checks(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    prompt_rows = [
        row
        for row in rows
        if row.get("method") == PROMPT_METHOD and int(row.get("seed", -1)) == 42
    ]
    by_class = {str(row.get("class_name")): row for row in prompt_rows}
    required = set(PROMPT_TARGET_CLASS_IDS)
    complete = set(by_class) == required
    loss_history = prompt_rows[0].get("loss_history", []) if prompt_rows else []
    losses = np.asarray(loss_history, dtype=float)
    finite_decreasing_loss = bool(
        losses.ndim == 1
        and losses.size >= 2
        and np.isfinite(losses).all()
        and losses[-1] < losses[0]
    )
    nonconstant_predictions = complete and all(
        bool(by_class[class_name].get("prediction_nonconstant"))
        for class_name in required
    )
    prompt_dependent = complete
    for class_name in required:
        values = np.asarray(
            by_class.get(class_name, {}).get(
                "prompt_probability_change_by_sample", []
            ),
            dtype=float,
        )
        prompt_dependent = bool(
            prompt_dependent
            and values.ndim == 1
            and values.size > 0
            and np.isfinite(values).all()
            and np.max(values) > 0.0
        )
    checkpoint_reproduced = complete and all(
        by_class[class_name].get("checkpoint_reproduced") is True
        for class_name in required
    )
    checks = {
        "finite_decreasing_loss": finite_decreasing_loss,
        "nonconstant_predictions": nonconstant_predictions,
        "prompt_dependent_probability_maps": prompt_dependent,
        "checkpoint_reproduced": checkpoint_reproduced,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "loss_first": float(losses[0]) if losses.size else None,
        "loss_last": float(losses[-1]) if losses.size else None,
    }


def _train_one_epoch(
    trainer,
    loader,
    prompt_config: PromptConfig,
    weights: dict[str, float],
    seed: int,
    method: str,
    *,
    empty_target_cap: float = 0.25,
):
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for images, masks in loader:
        batch = sample_prompt_batch(
            masks,
            prompt_config,
            generator=generator,
            empty_cap=empty_target_cap,
        )
        conditions = batch.prompts if method == PROMPT_METHOD else batch.class_ids
        positive_weights = _positive_weights_for_batch(batch.class_names, weights, trainer.device)
        losses.append(trainer.train_step(images, conditions, batch.targets, positive_weights))
    trainer.scheduler.step()
    return float(sum(losses) / max(1, len(losses)))


def _probe_condition(
    method: str, class_name: str, prompt_config: PromptConfig
) -> list[str] | torch.Tensor:
    if method == PROMPT_METHOD:
        return [prompt_config.classes[class_name].training[0]]
    if method == BASELINE_METHOD:
        return torch.tensor(
            [PROMPT_TARGET_CLASS_IDS[class_name]], dtype=torch.long
        )
    raise ValueError(f"unsupported GeoVLM method: {method}")


def _evaluate_probe(
    trainer,
    train,
    probe_indices_by_class,
    prompt_config: PromptConfig,
    weights: dict[str, float],
    method: str,
) -> dict[str, Any]:
    class_names = tuple(PROMPT_TARGET_CLASS_IDS)
    classes = {}
    losses = []
    class_changes = []
    finite = True
    for class_index, class_name in enumerate(class_names):
        wrong_name = class_names[(class_index + 1) % len(class_names)]
        correct_condition = _probe_condition(method, class_name, prompt_config)
        wrong_condition = _probe_condition(method, wrong_name, prompt_config)
        probability_min = None
        probability_max = None
        probability_changes = []
        for source_index in probe_indices_by_class[class_name]:
            image, mask = train[source_index]
            target = multiclass_to_binary(
                mask, PROMPT_TARGET_CLASS_IDS[class_name]
            ).unsqueeze(0)
            correct = trainer.predict(image.unsqueeze(0), correct_condition)
            wrong = trainer.predict(image.unsqueeze(0), wrong_condition)
            loss = trainer.criterion(
                correct,
                target.to(trainer.device),
                torch.tensor(
                    [weights[class_name]],
                    dtype=torch.float32,
                    device=trainer.device,
                ),
            )
            correct_probability = correct.sigmoid().detach().cpu()
            wrong_probability = wrong.sigmoid().detach().cpu()
            change = (correct_probability - wrong_probability).abs().mean()
            finite = bool(
                finite
                and torch.isfinite(correct).all()
                and torch.isfinite(wrong).all()
                and torch.isfinite(loss).all()
                and torch.isfinite(change).all()
            )
            losses.append(float(loss.detach().cpu()))
            probability_changes.append(float(change))
            sample_min = float(correct_probability.min())
            sample_max = float(correct_probability.max())
            probability_min = (
                sample_min if probability_min is None else min(probability_min, sample_min)
            )
            probability_max = (
                sample_max if probability_max is None else max(probability_max, sample_max)
            )

        prediction_range = (
            0.0
            if probability_min is None or probability_max is None
            else probability_max - probability_min
        )
        mean_change = (
            sum(probability_changes) / len(probability_changes)
            if probability_changes
            else 0.0
        )
        class_changes.append(mean_change)
        classes[class_name] = {
            "prediction_range": float(prediction_range),
            "prediction_nonconstant": bool(prediction_range > 0.0),
            "mean_prompt_probability_change": float(mean_change),
            "prompt_map_changed": bool(mean_change > 0.0),
        }

    return {
        "finite": bool(finite),
        "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "nonconstant_class_count": int(
            sum(value["prediction_nonconstant"] for value in classes.values())
        ),
        "prompt_map_changed_class_count": int(
            sum(value["prompt_map_changed"] for value in classes.values())
        ),
        "mean_prompt_probability_change": float(
            sum(class_changes) / len(class_changes)
        ),
        "classes": classes,
    }


def probe_rank(probe: dict[str, Any]) -> tuple[int, int, int, float, float]:
    if not probe.get("finite", False):
        return (0, 0, 0, float("-inf"), float("-inf"))
    return (
        1,
        int(probe["nonconstant_class_count"]),
        int(probe["prompt_map_changed_class_count"]),
        float(probe["mean_prompt_probability_change"]),
        -float(probe["mean_loss"]),
    )


def _all_prompt_values(prompt_config: PromptConfig, class_name: str, split: str):
    prompt_class = prompt_config.classes[class_name]
    return prompt_class.training if split == "seen" else prompt_class.held_out


def _predict_prob(trainer, image, condition):
    start = time.perf_counter()
    logits = trainer.predict(image.unsqueeze(0), condition)
    elapsed = time.perf_counter() - start
    return logits.sigmoid()[0].detach().cpu(), elapsed


def _write_preview(path, image, target, probability, threshold):
    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    target_rgb = np.repeat(np.asarray(target, dtype=np.uint8)[..., None] * 255, 3, axis=2)
    probability_rgb = np.repeat(
        np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    prediction_rgb = np.repeat(
        (np.asarray(probability) >= threshold).astype(np.uint8)[..., None] * 255,
        3,
        axis=2,
    )
    separator = np.full((rgb.shape[0], 2, 3), 255, dtype=np.uint8)
    panel = np.concatenate(
        (rgb, separator, target_rgb, separator, probability_rgb, separator, prediction_rgb),
        axis=1,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel, mode="RGB").save(path)
    return str(path)


def _evaluate_method(
    trainer,
    validation,
    prompt_config: PromptConfig,
    method: str,
    seed: int,
    threshold: float,
    trainable_params: int,
    frozen_params: int,
    preview_dir: str | Path,
    preview_count: int,
):
    rows = []
    class_names = list(PROMPT_TARGET_CLASS_IDS)
    base_preview_quota, extra_previews = divmod(max(0, preview_count), len(class_names))
    for class_index, (class_name, class_id) in enumerate(PROMPT_TARGET_CLASS_IDS.items()):
        seen_ious, held_ious, seen_dice, held_dice = [], [], [], []
        correct_ious, wrong_ious, probability_changes = [], [], []
        target_foreground, empties, latencies = [], [], []
        predicted_positive = 0
        predicted_pixels = 0
        prediction_nonconstant = False
        preview_paths = []
        preview_quota = base_preview_quota + int(class_index < extra_previews)
        for index in range(len(validation)):
            image, mask = validation[index]
            target = multiclass_to_binary(mask, class_id).bool().numpy()
            target_foreground.append(float(target.mean()))
            empties.append(float(not target.any()))
            if method == PROMPT_METHOD:
                predictions = {}
                for split in ("seen", "held_out"):
                    for prompt in _all_prompt_values(prompt_config, class_name, split):
                        probability, elapsed = _predict_prob(trainer, image, [prompt])
                        predictions[prompt] = probability.numpy()
                        latencies.append(elapsed)
                seen_values = [
                    binary_metrics(target, predictions[prompt] >= threshold)
                    for prompt in _all_prompt_values(prompt_config, class_name, "seen")
                ]
                held_values = [
                    binary_metrics(target, predictions[prompt] >= threshold)
                    for prompt in _all_prompt_values(prompt_config, class_name, "held_out")
                ]
                for probability in predictions.values():
                    prediction = probability >= threshold
                    predicted_positive += int(prediction.sum())
                    predicted_pixels += int(prediction.size)
                    prediction_nonconstant = bool(
                        prediction_nonconstant
                        or (prediction.any() and not prediction.all())
                    )
                seen_ious.append(sum(item["foreground_iou"] for item in seen_values) / len(seen_values))
                held_ious.append(sum(item["foreground_iou"] for item in held_values) / len(held_values))
                seen_dice.append(sum(item["dice"] for item in seen_values) / len(seen_values))
                held_dice.append(sum(item["dice"] for item in held_values) / len(held_values))
                correct_prompts = list(predictions)
                correct_ious.append(
                    sum(binary_metrics(target, predictions[p] >= threshold)["foreground_iou"] for p in correct_prompts)
                    / len(correct_prompts)
                )
                wrong_predictions = []
                for other_name in PROMPT_TARGET_CLASS_IDS:
                    if other_name == class_name:
                        continue
                    for prompt in _all_prompt_values(prompt_config, other_name, "seen") + _all_prompt_values(prompt_config, other_name, "held_out"):
                        probability, elapsed = _predict_prob(trainer, image, [prompt])
                        wrong_predictions.append(probability.numpy())
                        latencies.append(elapsed)
                wrong_ious.append(
                    sum(binary_metrics(target, probability >= threshold)["foreground_iou"] for probability in wrong_predictions)
                    / len(wrong_predictions)
                )
                correct_probability = sum(predictions.values()) / len(predictions)
                wrong_probability = sum(wrong_predictions) / len(wrong_predictions)
                probability_changes.append(float(abs(correct_probability - wrong_probability).mean()))
                for split in ("seen", "held_out"):
                    if len(preview_paths) >= preview_quota:
                        break
                    prompt = _all_prompt_values(prompt_config, class_name, split)[0]
                    preview_path = Path(preview_dir) / (
                        f"seed{seed}__{method}__{class_name}__{split}__{index:05d}.png"
                    )
                    preview_paths.append(
                        _write_preview(
                            preview_path,
                            image,
                            target,
                            predictions[prompt],
                            threshold,
                        )
                    )
            else:
                probability, elapsed = _predict_prob(trainer, image, torch.tensor([class_id]))
                latencies.append(elapsed)
                prediction = probability.numpy() >= threshold
                predicted_positive += int(prediction.sum())
                predicted_pixels += int(prediction.size)
                prediction_nonconstant = bool(
                    prediction_nonconstant or (prediction.any() and not prediction.all())
                )
                metrics = binary_metrics(target, prediction)
                seen_ious.append(metrics["foreground_iou"])
                held_ious.append(metrics["foreground_iou"])
                seen_dice.append(metrics["dice"])
                held_dice.append(metrics["dice"])
                if len(preview_paths) < preview_quota:
                    preview_path = Path(preview_dir) / (
                        f"seed{seed}__{method}__{class_name}__baseline__{index:05d}.png"
                    )
                    preview_paths.append(
                        _write_preview(
                            preview_path,
                            image,
                            target,
                            probability.numpy(),
                            threshold,
                        )
                    )

        row = {
            "method": method,
            "seed": int(seed),
            "class_name": class_name,
            "seen_iou": float(sum(seen_ious) / len(seen_ious)),
            "held_out_iou": float(sum(held_ious) / len(held_ious)),
            "seen_dice": float(sum(seen_dice) / len(seen_dice)),
            "held_out_dice": float(sum(held_dice) / len(held_dice)),
            "foreground_share": float(predicted_positive / predicted_pixels),
            "target_foreground_share": float(
                sum(target_foreground) / len(target_foreground)
            ),
            "empty_mask_rate": float(sum(empties) / len(empties)),
            "inference_latency_seconds": float(sum(latencies) / len(latencies)),
            "trainable_params": int(trainable_params),
            "frozen_params": int(frozen_params),
            "prediction_nonconstant": prediction_nonconstant,
            "preview_paths": preview_paths,
        }
        if method == PROMPT_METHOD:
            row.update(
                {
                    "correct_iou_by_sample": [float(value) for value in correct_ious],
                    "wrong_iou_by_sample": [float(value) for value in wrong_ious],
                    "prompt_probability_change_by_sample": [float(value) for value in probability_changes],
                }
            )
        rows.append(row)
    return rows


def _checkpoint_reproduces(
    config,
    method,
    seed,
    checkpoint_path,
    trainer,
    model_builder,
    validation,
    prompt_config,
    device,
):
    clone = model_builder(config, method, device)
    clone_trainer = build_trainer(clone, config, device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    clone_trainer.load_state_dict(state)
    image, mask = validation[0]
    if method == PROMPT_METHOD:
        condition = [prompt_config.classes["building"].training[0]]
    else:
        condition = torch.tensor([1])
    before = trainer.predict(image.unsqueeze(0), condition)
    after = clone_trainer.predict(image.unsqueeze(0), condition)
    return bool(torch.allclose(before, after, atol=1e-6))


def _run_pair(config, method, seed, train, validation, checkpoint_dir, preview_dir, device, model_builder):
    torch.manual_seed(seed)
    prompt_config = load_prompt_config(config["experiment"]["prompt_config"])
    weights = estimate_positive_weights(
        (train[index][1] for index in range(len(train))),
        clip=tuple(config["training"]["positive_weight_clip"]),
    )
    model = model_builder(config, method, device)
    trainer = build_trainer(model, config, device)
    loader = DataLoader(
        train,
        batch_size=int(config["experiment"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    losses = []
    checkpoint_path = Path(checkpoint_dir) / f"{method}__seed{seed}.pt"
    metadata = checkpoint_metadata(config, method, seed)
    total_epochs = int(config["experiment"]["epochs"])
    start_epoch = 0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        validate_checkpoint_metadata(state.get("metadata", {}), metadata)
        start_epoch, _ = trainer.load_state_dict(state)
        if not 0 <= start_epoch <= total_epochs:
            raise ValueError(
                f"checkpoint epoch {start_epoch} is outside 0..{total_epochs}"
            )
        losses = [float(value) for value in state.get("loss_history", [])]
    for epoch in range(start_epoch, total_epochs):
        losses.append(
            _train_one_epoch(
                trainer,
                loader,
                prompt_config,
                weights,
                seed + epoch,
                method,
                empty_target_cap=float(
                    config["experiment"]["empty_target_cap"]
                ),
            )
        )
        state = trainer.state_dict(epoch=epoch + 1, metadata=metadata)
        state["loss_history"] = losses
        _save_checkpoint(checkpoint_path, state)
    trainable_params, frozen_params = _parameter_counts(model)
    reproduced = _checkpoint_reproduces(
        config, method, seed, checkpoint_path, trainer, model_builder, validation, prompt_config, device
    )
    existing_previews = len(list(Path(preview_dir).glob(f"seed{seed}__*.png")))
    preview_count = max(
        0, int(config["evaluation"]["preview_count"]) - existing_previews
    )
    rows = _evaluate_method(
        trainer,
        validation,
        prompt_config,
        method,
        seed,
        float(config["evaluation"]["threshold"]),
        trainable_params,
        frozen_params,
        preview_dir,
        preview_count,
    )
    for row in rows:
        row["checkpoint_reproduced"] = reproduced
        row["loss_history"] = losses
        row["loss_first"] = losses[0] if losses else None
        row["loss_last"] = losses[-1] if losses else None
    return rows


def _raw_results_payload(rows):
    payload = {
        "schema": "paper12.geovlm_prompt_results.v1",
        "rows": rows,
    }
    if any(
        row.get("method") == PROMPT_METHOD and int(row.get("seed", -1)) == 42
        for row in rows
    ):
        payload["seed42_smoke"] = seed42_smoke_checks(rows)
    return payload


def run_experiment(
    config,
    *,
    output_path,
    summary_output_path,
    checkpoint_dir,
    preview_dir,
    stage="full",
    device="cpu",
    dataset_builder=build_datasets,
    model_builder=build_model,
):
    if config["experiment"].get("allow_synthetic_fallback") is not False:
        raise ValueError("GeoVLM runner requires allow_synthetic_fallback: false")
    output_path = Path(output_path)
    summary_output_path = Path(summary_output_path)
    rows = []
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    if stage == "seed42":
        pairs = [(PROMPT_METHOD, 42)]
    elif stage == "full":
        pairs = [
            (method, int(seed))
            for method in config["methods"]
            for seed in config["experiment"]["seeds"]
        ]
    else:
        raise ValueError("stage must be seed42 or full")
    done = completed_keys(rows)
    pending = [(method, seed) for method, seed in pairs if (method, seed) not in done]
    if any(
        row.get("method") == PROMPT_METHOD and int(row.get("seed", -1)) == 42
        for row in rows
    ):
        smoke = seed42_smoke_checks(rows)
        if not smoke["passed"]:
            raise RuntimeError(
                "seed42 smoke checks failed: " + ", ".join(smoke["failed_checks"])
            )
    if pending:
        train, validation = dataset_builder(config)
        for method, seed in pending:
            new_rows = _run_pair(
                config,
                method,
                seed,
                train,
                validation,
                checkpoint_dir,
                preview_dir,
                device,
                model_builder,
            )
            rows.extend(new_rows)
            raw_payload = _raw_results_payload(rows)
            _atomic_json(output_path, raw_payload)
            _atomic_json(
                summary_output_path,
                build_summary(
                    rows,
                    bootstrap_iterations=int(config["evaluation"]["bootstrap_iterations"]),
                    seed=seed,
                ),
            )
            if method == PROMPT_METHOD and seed == 42:
                smoke = raw_payload["seed42_smoke"]
                if not smoke["passed"]:
                    raise RuntimeError(
                        "seed42 smoke checks failed: "
                        + ", ".join(smoke["failed_checks"])
                    )
    elif rows:
        _atomic_json(
            summary_output_path,
            build_summary(
                rows,
                bootstrap_iterations=int(config["evaluation"]["bootstrap_iterations"]),
                seed=0,
            ),
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--preview-dir", required=True)
    parser.add_argument("--stage", choices=("seed42", "full"), default="full")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    run_experiment(
        load_config(args.config),
        output_path=args.output,
        summary_output_path=args.summary_output,
        checkpoint_dir=args.checkpoint_dir,
        preview_dir=args.preview_dir,
        stage=args.stage,
        device=args.device,
    )


if __name__ == "__main__":
    main()
