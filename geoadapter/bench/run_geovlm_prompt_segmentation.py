from __future__ import annotations

import copy
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
    prompt_batch_from_class_names,
    validate_landcoverai_mask,
)
from geoadapter.bench.geovlm_training import (
    AssignedPromptDataset,
    build_epoch_assignments,
    reserve_training_probe,
    scan_target_present_pool,
)
from geoadapter.engine.prompt_segmentation import (
    PromptSegmentationLoss,
    PromptSegmentationTrainer,
)
from geoadapter.models.prompt_segmentation import (
    PromptSegmentationModel,
    ThreeHeadSegmentationBaseline,
)
from geoadapter.bench.geovlm_prompt_summary import (
    REQUIRED_CLASSES,
    REQUIRED_METHODS,
    REQUIRED_SEEDS,
    binary_metrics,
    build_summary,
)


PROMPT_METHOD = "siglip_film_dense_similarity_houlsby"
BASELINE_METHOD = "no_text_three_binary_heads_houlsby"
TRAINING_CONTRACT = "paper12.geovlm_prompt_training.v2"


def _require_training_contract(config):
    actual = config["experiment"].get("training_contract")
    if actual != TRAINING_CONTRACT:
        raise ValueError(
            f"unsupported GeoVLM training contract: {actual!r}; "
            f"expected {TRAINING_CONTRACT!r}"
        )
    return actual


def _incompatible_result_artifact(path):
    return ValueError(
        f"incompatible GeoVLM result artifact {path}; "
        "archive it before recovery"
    )


def _rows_from_compatible_payload(
    payload,
    path,
    training_contract,
    expected_siglip_model_id,
    expected_siglip_revision,
):
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "paper12.geovlm_prompt_results.v2"
        or payload.get("training_contract") != training_contract
        or not isinstance(rows, list)
        or any(
            not isinstance(row, dict)
            or row.get("training_contract") != training_contract
            or row.get("siglip_model_id") != expected_siglip_model_id
            or row.get("siglip_revision") != expected_siglip_revision
            for row in rows
        )
    ):
        raise _incompatible_result_artifact(path)
    seen = set()
    classes_by_pair = {}
    for row in rows:
        method = row.get("method")
        seed = row.get("seed")
        class_name = row.get("class_name")
        if (
            method not in REQUIRED_METHODS
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in REQUIRED_SEEDS
            or class_name not in REQUIRED_CLASSES
        ):
            raise _incompatible_result_artifact(path)
        key = (method, seed, class_name)
        if key in seen:
            raise _incompatible_result_artifact(path)
        seen.add(key)
        classes_by_pair.setdefault((method, seed), set()).add(class_name)
    if any(
        classes != set(REQUIRED_CLASSES) for classes in classes_by_pair.values()
    ):
        raise _incompatible_result_artifact(path)
    return rows


def _preflight_checkpoint_layouts(checkpoint_dir, pairs):
    checkpoint_dir = Path(checkpoint_dir)
    for method, seed in pairs:
        checkpoint_base = checkpoint_dir / f"{method}__seed{seed}"
        last_checkpoint_path = checkpoint_base.with_suffix(".last.pt")
        best_checkpoint_path = checkpoint_base.with_suffix(".best.pt")
        legacy_checkpoint_path = checkpoint_base.with_suffix(".pt")
        if legacy_checkpoint_path.exists():
            raise ValueError(
                f"legacy checkpoint exists at {legacy_checkpoint_path}; "
                "archive it before recovery"
            )
        if best_checkpoint_path.exists() and not last_checkpoint_path.exists():
            raise ValueError("best checkpoint exists without last checkpoint")
        if last_checkpoint_path.exists() and not best_checkpoint_path.exists():
            raise ValueError("last checkpoint exists without best checkpoint")


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
        "positive_weight_policy": "full_source_training_split_v1",
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
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
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
        "positive_weight_policy",
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
):
    generator = torch.Generator().manual_seed(seed)
    loss_sum = 0.0
    stats = {
        "sample_count": 0,
        "empty_target_count": 0,
        "prompt_counts": {name: 0 for name in PROMPT_TARGET_CLASS_IDS},
        "nonempty_prompt_counts": {
            name: 0 for name in PROMPT_TARGET_CLASS_IDS
        },
    }
    for images, masks, class_names in loader:
        batch = prompt_batch_from_class_names(
            masks,
            class_names,
            prompt_config,
            generator=generator,
        )
        conditions = batch.prompts if method == PROMPT_METHOD else batch.class_ids
        positive_weights = _positive_weights_for_batch(batch.class_names, weights, trainer.device)
        batch_size = len(batch.class_names)
        batch_loss = trainer.train_step(
            images, conditions, batch.targets, positive_weights
        )
        loss_sum += batch_loss * batch_size
        stats["sample_count"] += batch_size
        stats["empty_target_count"] += batch.empty_count
        nonempty = batch.targets.flatten(1).sum(dim=1).ne(0).tolist()
        for class_name, target_is_nonempty in zip(batch.class_names, nonempty):
            stats["prompt_counts"][class_name] += 1
            stats["nonempty_prompt_counts"][class_name] += int(
                target_is_nonempty
            )
    trainer.scheduler.step()
    return {
        "loss": float(loss_sum / max(1, stats["sample_count"])),
        **stats,
    }


def _merge_training_stats(total, epoch):
    total["sample_count"] += int(epoch["sample_count"])
    total["empty_target_count"] += int(epoch["empty_target_count"])
    for class_name in PROMPT_TARGET_CLASS_IDS:
        total["prompt_counts"][class_name] += int(
            epoch["prompt_counts"][class_name]
        )
        total["nonempty_prompt_counts"][class_name] += int(
            epoch["nonempty_prompt_counts"][class_name]
        )
    return total


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
    actual_names = set(probe_indices_by_class)
    expected_names = set(class_names)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            "probe_indices_by_class must contain exactly "
            f"{list(class_names)}; missing={missing}; unexpected={unexpected}"
        )
    for class_name in class_names:
        if not probe_indices_by_class[class_name]:
            raise ValueError(f"probe indices for {class_name} must be non-empty")

    classes = {}
    losses = []
    class_changes = []
    finite = True
    was_training = trainer.model.training
    try:
        for class_index, class_name in enumerate(class_names):
            wrong_name = class_names[(class_index + 1) % len(class_names)]
            correct_condition = _probe_condition(method, class_name, prompt_config)
            wrong_condition = _probe_condition(method, wrong_name, prompt_config)
            spatial_ranges = []
            probability_changes = []
            class_finite = True
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
                sample_finite = bool(
                    torch.isfinite(correct).all()
                    and torch.isfinite(wrong).all()
                    and torch.isfinite(loss).all()
                    and torch.isfinite(change).all()
                )
                if not sample_finite:
                    finite = False
                    class_finite = False
                    continue

                losses.append(float(loss.detach().cpu()))
                probability_changes.append(float(change))
                spatial_ranges.append(
                    float(correct_probability.max() - correct_probability.min())
                )

            if class_finite:
                prediction_range = max(spatial_ranges)
                mean_change = sum(probability_changes) / len(probability_changes)
                class_changes.append(mean_change)
                classes[class_name] = {
                    "prediction_range": float(prediction_range),
                    "prediction_nonconstant": bool(prediction_range > 0.0),
                    "mean_prompt_probability_change": float(mean_change),
                    "prompt_map_changed": bool(mean_change > 0.0),
                }
            else:
                classes[class_name] = {
                    "prediction_range": None,
                    "prediction_nonconstant": False,
                    "mean_prompt_probability_change": None,
                    "prompt_map_changed": False,
                }
    finally:
        trainer.model.train(was_training)

    return {
        "finite": bool(finite),
        "mean_loss": float(sum(losses) / len(losses)) if finite else None,
        "nonconstant_class_count": int(
            sum(value["prediction_nonconstant"] for value in classes.values())
        ),
        "prompt_map_changed_class_count": int(
            sum(value["prompt_map_changed"] for value in classes.values())
        ),
        "mean_prompt_probability_change": (
            float(sum(class_changes) / len(class_changes)) if finite else None
        ),
        "classes": classes,
    }


def probe_rank(probe: dict[str, Any]) -> tuple[int, int, int, float, float]:
    if not probe.get("finite", False):
        return (0, 0, 0, 0.0, 0.0)
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
    comparisons = []
    for class_name in PROMPT_TARGET_CLASS_IDS:
        condition = _probe_condition(method, class_name, prompt_config)
        before = trainer.predict(image.unsqueeze(0), condition)
        after = clone_trainer.predict(image.unsqueeze(0), condition)
        comparisons.append(torch.allclose(before, after, atol=1e-6))
    return bool(all(comparisons))


def _empty_training_stats():
    return {
        "sample_count": 0,
        "empty_target_count": 0,
        "prompt_counts": {name: 0 for name in PROMPT_TARGET_CLASS_IDS},
        "nonempty_prompt_counts": {
            name: 0 for name in PROMPT_TARGET_CLASS_IDS
        },
    }


def _normalize_training_stats(stats):
    if not isinstance(stats, dict):
        raise ValueError("checkpoint training_stats must be a dictionary")

    def count_value(value, field):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"checkpoint {field} must be a nonnegative integer")
        value = int(value)
        if value < 0:
            raise ValueError(f"checkpoint {field} must be a nonnegative integer")
        return value

    normalized = {
        "sample_count": count_value(
            stats.get("sample_count"), "training_stats.sample_count"
        ),
        "empty_target_count": count_value(
            stats.get("empty_target_count"),
            "training_stats.empty_target_count",
        ),
    }
    expected_classes = set(PROMPT_TARGET_CLASS_IDS)
    for field in ("prompt_counts", "nonempty_prompt_counts"):
        values = stats.get(field)
        if not isinstance(values, dict) or set(values) != expected_classes:
            raise ValueError(
                f"checkpoint training_stats.{field} must contain exactly "
                f"{list(PROMPT_TARGET_CLASS_IDS)}"
            )
        normalized[field] = {
            class_name: count_value(
                values[class_name], f"training_stats.{field}.{class_name}"
            )
            for class_name in PROMPT_TARGET_CLASS_IDS
        }

    if normalized["empty_target_count"] > normalized["sample_count"]:
        raise ValueError("checkpoint empty target count exceeds sample count")
    if sum(normalized["prompt_counts"].values()) != normalized["sample_count"]:
        raise ValueError("checkpoint prompt counts do not sum to sample count")
    if sum(normalized["nonempty_prompt_counts"].values()) != (
        normalized["sample_count"] - normalized["empty_target_count"]
    ):
        raise ValueError(
            "checkpoint nonempty prompt counts do not match empty target count"
        )
    return normalized


def _plain_probe_indices(probe_indices_by_class):
    return {
        class_name: [int(index) for index in probe_indices_by_class[class_name]]
        for class_name in PROMPT_TARGET_CLASS_IDS
    }


def _normalize_checkpoint_probe_indices(value):
    if not isinstance(value, dict) or set(value) != set(PROMPT_TARGET_CLASS_IDS):
        raise ValueError(
            "checkpoint probe_indices_by_class must contain exactly "
            f"{list(PROMPT_TARGET_CLASS_IDS)}"
        )
    normalized = {}
    for class_name in PROMPT_TARGET_CLASS_IDS:
        indices = value[class_name]
        if not isinstance(indices, (list, tuple)) or not indices:
            raise ValueError(
                f"checkpoint probe indices for {class_name} must be a non-empty list"
            )
        if any(
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            for index in indices
        ):
            raise ValueError("checkpoint probe indices must be integers")
        normalized[class_name] = [int(index) for index in indices]
    return normalized


def _validate_checkpoint_positive_weights(state, expected):
    actual = state.get("positive_weights")
    if not isinstance(actual, dict) or set(actual) != set(PROMPT_TARGET_CLASS_IDS):
        raise ValueError(
            "checkpoint positive weights mismatch for full source training split; "
            "archive checkpoints and restart recovery"
        )
    actual = {name: float(actual[name]) for name in PROMPT_TARGET_CLASS_IDS}
    if actual != expected or not np.isfinite(
        np.asarray(list(actual.values()), dtype=float)
    ).all():
        raise ValueError(
            "checkpoint positive weights mismatch for full source training split; "
            "archive checkpoints and restart recovery"
        )


def _validate_checkpoint_training_sample_count(
    training_stats, checkpoint_epoch, epoch_sample_count
):
    expected = checkpoint_epoch * epoch_sample_count
    if training_stats["sample_count"] != expected:
        raise ValueError(
            "checkpoint training sample count mismatch: expected "
            f"checkpoint epoch {checkpoint_epoch} * split size "
            f"{epoch_sample_count} = {expected}, got "
            f"{training_stats['sample_count']}"
        )


def _validate_checkpoint_role(state, expected_role):
    if state.get("checkpoint_role") != expected_role:
        raise ValueError(
            f"{expected_role} checkpoint role mismatch: expected "
            f"{expected_role!r}, got {state.get('checkpoint_role')!r}"
        )


def validate_checkpoint_best_selection(state, checkpoint_epoch):
    if isinstance(checkpoint_epoch, bool) or not isinstance(
        checkpoint_epoch, (int, np.integer)
    ):
        raise ValueError("checkpoint epoch must be an integer")
    checkpoint_epoch = int(checkpoint_epoch)
    if checkpoint_epoch <= 0:
        raise ValueError("checkpoint epoch must be positive")
    loss_history = state.get("loss_history", [])
    probe_history = state.get("probe_history", [])
    if not isinstance(loss_history, (list, tuple)):
        raise ValueError("checkpoint loss_history must be a list or tuple")
    if not isinstance(probe_history, (list, tuple)):
        raise ValueError("checkpoint probe_history must be a list or tuple")
    try:
        losses = [float(value) for value in loss_history]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("checkpoint loss_history values must be numeric") from exc
    if len(losses) != checkpoint_epoch:
        raise ValueError("checkpoint loss_history length must equal checkpoint epoch")
    if len(probe_history) != checkpoint_epoch or not all(
        isinstance(probe, dict) for probe in probe_history
    ):
        raise ValueError("checkpoint probe_history length must equal checkpoint epoch")
    if not np.isfinite(np.asarray(losses, dtype=float)).all():
        raise ValueError("checkpoint loss_history must be finite")
    for index, (loss, probe) in enumerate(zip(losses, probe_history), start=1):
        if probe.get("epoch") != index:
            raise ValueError("checkpoint probe_history epochs must be consecutive")
        try:
            probe_training_loss = float(
                probe.get("training_loss", float("nan"))
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "checkpoint probe_history training_loss must be numeric"
            ) from exc
        if probe_training_loss != loss:
            raise ValueError(
                "checkpoint probe training loss must match loss_history"
            )

    best_epoch = state.get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(
        best_epoch, (int, np.integer)
    ):
        raise ValueError("checkpoint best_epoch must be an integer")
    best_epoch = int(best_epoch)
    if not 1 <= best_epoch <= checkpoint_epoch:
        raise ValueError("checkpoint best_epoch is outside completed history")
    best_rank_value = state.get("best_probe_rank")
    if not isinstance(best_rank_value, (list, tuple)) or len(best_rank_value) != 5:
        raise ValueError("checkpoint best_probe_rank must contain five values")
    best_rank = tuple(best_rank_value)
    try:
        best_rank_array = np.asarray(best_rank, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "checkpoint best_probe_rank values must be numeric"
        ) from exc
    if not np.isfinite(best_rank_array).all():
        raise ValueError("checkpoint best_probe_rank must be finite")
    expected_best_epoch = None
    expected_best_rank = None
    for epoch, probe in enumerate(probe_history, start=1):
        try:
            rank = probe_rank(probe)
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise ValueError(
                "checkpoint probe_history rank fields are invalid"
            ) from exc
        if expected_best_rank is None or rank > expected_best_rank:
            expected_best_epoch = epoch
            expected_best_rank = rank
    if best_epoch != expected_best_epoch or best_rank != expected_best_rank:
        raise ValueError("checkpoint best selection mismatch")
    metadata = state.get("metadata", {})
    if metadata.get("best_epoch") != best_epoch:
        raise ValueError(
            "checkpoint best selection mismatch: metadata best_epoch"
        )
    if metadata.get("best_probe_rank") != list(best_rank):
        raise ValueError(
            "checkpoint best selection mismatch: metadata best_probe_rank"
        )
    return losses, list(probe_history), best_epoch, best_rank


def _restore_checkpoint_history(
    state,
    *,
    checkpoint_epoch,
    total_epochs,
    probe_indices_by_class,
    probe_sha256,
):
    losses, probe_history, best_epoch, best_rank = (
        validate_checkpoint_best_selection(state, checkpoint_epoch)
    )
    if checkpoint_epoch > total_epochs:
        raise ValueError(
            f"checkpoint epoch {checkpoint_epoch} is outside 1..{total_epochs}"
        )

    stored_probe_indices = _normalize_checkpoint_probe_indices(
        state.get("probe_indices_by_class")
    )
    if stored_probe_indices != probe_indices_by_class:
        raise ValueError("checkpoint probe split does not match current dataset")
    if state.get("probe_sha256") != probe_sha256:
        raise ValueError("checkpoint probe_sha256 does not match current split")

    training_stats = _normalize_training_stats(state.get("training_stats"))
    return losses, list(probe_history), training_stats, best_epoch, best_rank


def _load_checkpoint_state(path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"incompatible GeoVLM checkpoint {path}; "
            "archive it before recovery"
        ) from exc
    if not isinstance(state, dict):
        raise ValueError(
            f"incompatible GeoVLM checkpoint {path}; "
            "archive it before recovery"
        )
    return state


def _validate_resume_checkpoint_pair(
    config,
    method,
    seed,
    checkpoint_dir,
    *,
    weights,
    split,
    device,
):
    checkpoint_base = Path(checkpoint_dir) / f"{method}__seed{seed}"
    last_checkpoint_path = checkpoint_base.with_suffix(".last.pt")
    best_checkpoint_path = checkpoint_base.with_suffix(".best.pt")
    total_epochs = int(config["experiment"]["epochs"])
    probe_indices_by_class = _plain_probe_indices(
        split.probe_indices_by_class
    )
    metadata = checkpoint_metadata(config, method, seed)

    state = _load_checkpoint_state(last_checkpoint_path, device)
    _validate_checkpoint_role(state, "last")
    validate_checkpoint_metadata(state.get("metadata", {}), metadata)
    _validate_checkpoint_positive_weights(state, weights)
    checkpoint_epoch = state.get("epoch")
    (
        losses,
        probe_history,
        training_stats,
        best_epoch,
        best_rank,
    ) = _restore_checkpoint_history(
        state,
        checkpoint_epoch=checkpoint_epoch,
        total_epochs=total_epochs,
        probe_indices_by_class=probe_indices_by_class,
        probe_sha256=split.probe_sha256,
    )
    _validate_checkpoint_training_sample_count(
        training_stats, checkpoint_epoch, len(split.training_samples)
    )

    best_state = _load_checkpoint_state(best_checkpoint_path, device)
    _validate_checkpoint_role(best_state, "best")
    validate_checkpoint_metadata(best_state.get("metadata", {}), metadata)
    _validate_checkpoint_positive_weights(best_state, weights)
    best_state_epoch = best_state.get("epoch")
    (
        best_losses,
        best_probe_history,
        best_training_stats,
        stored_best_epoch,
        stored_best_rank,
    ) = _restore_checkpoint_history(
        best_state,
        checkpoint_epoch=best_state_epoch,
        total_epochs=total_epochs,
        probe_indices_by_class=probe_indices_by_class,
        probe_sha256=split.probe_sha256,
    )
    _validate_checkpoint_training_sample_count(
        best_training_stats,
        best_state_epoch,
        len(split.training_samples),
    )
    if best_state_epoch != best_epoch:
        raise ValueError("best checkpoint epoch does not match last checkpoint")
    if stored_best_epoch != best_epoch or stored_best_rank != best_rank:
        raise ValueError("best checkpoint references do not match last checkpoint")
    if best_losses != losses[:best_epoch]:
        raise ValueError("best checkpoint history mismatch: loss_history")
    if best_probe_history != probe_history[:best_epoch]:
        raise ValueError("best checkpoint history mismatch: probe_history")
    return {
        "last_state": state,
        "best_state": best_state,
        "checkpoint_epoch": int(checkpoint_epoch),
        "losses": losses,
        "probe_history": probe_history,
        "training_stats": training_stats,
        "best_epoch": best_epoch,
        "best_rank": best_rank,
    }


def _preflight_existing_checkpoint_contents(
    config, pairs, train, checkpoint_dir, device, model_builder
):
    weights = estimate_positive_weights(
        (train[index][1] for index in range(len(train))),
        clip=tuple(config["training"]["positive_weight_clip"]),
    )
    pool = scan_target_present_pool(train)
    for method, seed in pairs:
        checkpoint_base = Path(checkpoint_dir) / f"{method}__seed{seed}"
        if not checkpoint_base.with_suffix(".last.pt").exists():
            continue
        split = reserve_training_probe(
            pool,
            seed=seed,
            positives_per_class=int(
                config["experiment"]["probe_positives_per_class"]
            ),
        )
        try:
            resume = _validate_resume_checkpoint_pair(
                config,
                method,
                seed,
                checkpoint_dir,
                weights=weights,
                split=split,
                device=device,
            )
            model = model_builder(config, method, device)
            trainer = build_trainer(model, config, device)
            for role in ("last", "best"):
                state = resume[f"{role}_state"]
                loaded_epoch, _ = trainer.load_state_dict(state)
                if loaded_epoch != state["epoch"]:
                    raise ValueError(
                        f"{role} checkpoint epoch changed during preflight load"
                    )
        except Exception as exc:
            raise ValueError(
                f"incompatible GeoVLM checkpoint pair {method}/seed{seed}; "
                f"archive it before recovery: {exc}"
            ) from exc


def _run_pair(config, method, seed, train, validation, checkpoint_dir, preview_dir, device, model_builder):
    checkpoint_base = Path(checkpoint_dir) / f"{method}__seed{seed}"
    last_checkpoint_path = checkpoint_base.with_suffix(".last.pt")
    best_checkpoint_path = checkpoint_base.with_suffix(".best.pt")
    legacy_checkpoint_path = checkpoint_base.with_suffix(".pt")
    if legacy_checkpoint_path.exists():
        raise ValueError(
            f"legacy checkpoint exists at {legacy_checkpoint_path}; "
            "archive it before recovery"
        )
    if best_checkpoint_path.exists() and not last_checkpoint_path.exists():
        raise ValueError("best checkpoint exists without last checkpoint")
    if last_checkpoint_path.exists() and not best_checkpoint_path.exists():
        raise ValueError("last checkpoint exists without best checkpoint")

    total_epochs = int(config["experiment"]["epochs"])
    if total_epochs <= 0:
        raise ValueError("GeoVLM training epochs must be positive")
    torch.manual_seed(seed)
    prompt_config = load_prompt_config(config["experiment"]["prompt_config"])
    weights = estimate_positive_weights(
        (train[index][1] for index in range(len(train))),
        clip=tuple(config["training"]["positive_weight_clip"]),
    )
    pool = scan_target_present_pool(train)
    split = reserve_training_probe(
        pool,
        seed=seed,
        positives_per_class=int(
            config["experiment"]["probe_positives_per_class"]
        ),
    )
    probe_indices_by_class = _plain_probe_indices(
        split.probe_indices_by_class
    )
    losses = []
    probe_history = []
    training_stats = _empty_training_stats()
    best_rank = None
    best_epoch = None
    metadata = checkpoint_metadata(config, method, seed)
    start_epoch = 0
    resume = None
    if last_checkpoint_path.exists():
        resume = _validate_resume_checkpoint_pair(
            config,
            method,
            seed,
            checkpoint_dir,
            weights=weights,
            split=split,
            device=device,
        )
        start_epoch = resume["checkpoint_epoch"]
        losses = resume["losses"]
        probe_history = resume["probe_history"]
        training_stats = resume["training_stats"]
        best_epoch = resume["best_epoch"]
        best_rank = resume["best_rank"]

    model = model_builder(config, method, device)
    trainer = build_trainer(model, config, device)
    if resume is not None:
        loaded_epoch, _ = trainer.load_state_dict(resume["last_state"])
        if loaded_epoch != start_epoch:
            raise ValueError("last checkpoint epoch changed during model reload")

    batch_size = int(config["experiment"]["batch_size"])
    empty_target_cap = float(config["experiment"]["empty_target_cap"])
    for epoch in range(start_epoch, total_epochs):
        assignments = build_epoch_assignments(
            split,
            batch_size=batch_size,
            empty_target_cap=empty_target_cap,
            seed=seed + epoch,
        )
        loader = DataLoader(
            AssignedPromptDataset(train, assignments),
            batch_size=batch_size,
            shuffle=False,
        )
        epoch_stats = _train_one_epoch(
            trainer,
            loader,
            prompt_config,
            weights,
            seed + epoch,
            method,
        )
        loss = float(epoch_stats["loss"])
        if not np.isfinite(loss):
            raise ValueError("training loss must be finite")
        losses.append(loss)
        epoch_counts = _normalize_training_stats(
            {
                key: epoch_stats[key]
                for key in (
                    "sample_count",
                    "empty_target_count",
                    "prompt_counts",
                    "nonempty_prompt_counts",
                )
            }
        )
        if epoch_counts["sample_count"] != len(split.training_samples):
            raise ValueError(
                "training epoch sample count mismatch: expected split size "
                f"{len(split.training_samples)}, got "
                f"{epoch_counts['sample_count']}"
            )
        _merge_training_stats(training_stats, epoch_counts)
        probe = _evaluate_probe(
            trainer,
            train,
            probe_indices_by_class,
            prompt_config,
            weights,
            method,
        )
        probe = {
            **probe,
            "epoch": epoch + 1,
            "training_loss": loss,
        }
        probe_history.append(probe)
        current_rank = probe_rank(probe)
        is_best = best_rank is None or current_rank > best_rank
        if is_best:
            best_rank = current_rank
            best_epoch = epoch + 1
        checkpoint_metadata_with_best = {
            **metadata,
            "best_epoch": best_epoch,
            "best_probe_rank": list(best_rank),
        }
        state = trainer.state_dict(
            epoch=epoch + 1,
            metadata=checkpoint_metadata_with_best,
        )
        state.update(
            {
                "loss_history": list(losses),
                "probe_history": list(probe_history),
                "training_stats": {
                    "sample_count": training_stats["sample_count"],
                    "empty_target_count": training_stats[
                        "empty_target_count"
                    ],
                    "prompt_counts": dict(training_stats["prompt_counts"]),
                    "nonempty_prompt_counts": dict(
                        training_stats["nonempty_prompt_counts"]
                    ),
                },
                "probe_indices_by_class": {
                    name: list(indices)
                    for name, indices in probe_indices_by_class.items()
                },
                "probe_sha256": split.probe_sha256,
                "positive_weights": dict(weights),
                "best_epoch": best_epoch,
                "best_probe_rank": list(best_rank),
            }
        )
        if is_best:
            _save_checkpoint(
                best_checkpoint_path, {**state, "checkpoint_role": "best"}
            )
        _save_checkpoint(
            last_checkpoint_path, {**state, "checkpoint_role": "last"}
        )

    if best_epoch is None or best_rank is None or not best_checkpoint_path.exists():
        raise RuntimeError("GeoVLM training did not produce a best checkpoint")
    best_state = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False
    )
    _validate_checkpoint_role(best_state, "best")
    validate_checkpoint_metadata(best_state.get("metadata", {}), metadata)
    _validate_checkpoint_positive_weights(best_state, weights)
    if best_state.get("probe_sha256") != split.probe_sha256:
        raise ValueError("best checkpoint probe_sha256 does not match current split")
    if int(best_state.get("epoch", -1)) != best_epoch:
        raise ValueError("best checkpoint epoch does not match selected epoch")
    final_best_training_stats = _normalize_training_stats(
        best_state.get("training_stats")
    )
    _validate_checkpoint_training_sample_count(
        final_best_training_stats,
        best_epoch,
        len(split.training_samples),
    )
    trainer.load_state_dict(best_state)
    trainable_params, frozen_params = _parameter_counts(model)
    reproduced = _checkpoint_reproduces(
        config,
        method,
        seed,
        best_checkpoint_path,
        trainer,
        model_builder,
        validation,
        prompt_config,
        device,
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
    observed_sample_count = training_stats["sample_count"]
    if observed_sample_count <= 0:
        raise ValueError("observed training sample count must be positive")
    selected_losses = losses[:best_epoch]
    best_probe = probe_history[best_epoch - 1]
    for row in rows:
        row.update(
            {
                "training_contract": metadata["training_contract"],
                "siglip_model_id": metadata["siglip_model_id"],
                "siglip_revision": metadata["siglip_revision"],
                "checkpoint_reproduced": reproduced,
                "source_training_size": pool.source_size,
                "target_present_pool_size": len(pool.samples),
                "excluded_no_target_count": pool.excluded_no_target_count,
                "excluded_no_target_share": float(
                    pool.excluded_no_target_share
                ),
                "probe_indices_by_class": {
                    name: list(indices)
                    for name, indices in probe_indices_by_class.items()
                },
                "probe_sha256": split.probe_sha256,
                "per_class_prompt_counts": dict(
                    training_stats["prompt_counts"]
                ),
                "per_class_nonempty_prompt_counts": dict(
                    training_stats["nonempty_prompt_counts"]
                ),
                "observed_empty_target_count": training_stats[
                    "empty_target_count"
                ],
                "observed_training_sample_count": observed_sample_count,
                "observed_empty_target_share": float(
                    training_stats["empty_target_count"]
                    / observed_sample_count
                ),
                "best_epoch": best_epoch,
                "best_probe_rank": list(best_rank),
                "best_probe": copy.deepcopy(best_probe),
                "full_loss_history": list(losses),
                "loss_history": list(selected_losses),
                "loss_first": selected_losses[0],
                "loss_last": selected_losses[-1],
            }
        )
    return rows


def _raw_results_payload(rows, training_contract):
    payload = {
        "schema": "paper12.geovlm_prompt_results.v2",
        "training_contract": training_contract,
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
    training_contract = _require_training_contract(config)
    if config["experiment"].get("allow_synthetic_fallback") is not False:
        raise ValueError("GeoVLM runner requires allow_synthetic_fallback: false")
    output_path = Path(output_path)
    summary_output_path = Path(summary_output_path)
    rows = []
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        rows = _rows_from_compatible_payload(
            payload,
            output_path,
            training_contract,
            config["text_encoder"]["model_id"],
            config["text_encoder"].get("revision"),
        )
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
    _preflight_checkpoint_layouts(checkpoint_dir, pairs)
    done = completed_keys(rows)
    pending = [(method, seed) for method, seed in pairs if (method, seed) not in done]
    existing_checkpoint_pairs = [
        (method, seed)
        for method, seed in pairs
        if (
            Path(checkpoint_dir) / f"{method}__seed{seed}"
        ).with_suffix(".last.pt").exists()
    ]
    if any(
        row.get("method") == PROMPT_METHOD and int(row.get("seed", -1)) == 42
        for row in rows
    ):
        smoke = seed42_smoke_checks(rows)
        if not smoke["passed"]:
            raise RuntimeError(
                "seed42 smoke checks failed: " + ", ".join(smoke["failed_checks"])
            )
    train = validation = None
    if pending or existing_checkpoint_pairs:
        train, validation = dataset_builder(config)
    if existing_checkpoint_pairs:
        _preflight_existing_checkpoint_contents(
            config,
            pairs,
            train,
            checkpoint_dir,
            device,
            model_builder,
        )
    if pending:
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
            raw_payload = _raw_results_payload(rows, training_contract)
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
