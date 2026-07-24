import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

from geoadapter.bench.run_geovlm_prompt_segmentation import (
    BASELINE_METHOD,
    PROMPT_METHOD,
    _atomic_json,
    _checkpoint_reproduces,
    _evaluate_probe,
    _probe_condition,
    _train_one_epoch,
    build_trainer,
    checkpoint_metadata,
    completed_keys,
    estimate_positive_weights,
    probe_rank,
    run_experiment,
    seed42_smoke_checks,
    sha256_file,
    validate_checkpoint_metadata,
)
from geoadapter.bench.geovlm_training import (
    reserve_training_probe,
    scan_target_present_pool,
)
from geoadapter.data.prompt_segmentation import load_prompt_config


CONFIG_PATH = Path("geoadapter/bench/configs/geovlm_prompt_segmentation.yaml")


def test_geovlm_prompt_config_is_real_data_only():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["training_contract"] == (
        "paper12.geovlm_prompt_training.v2"
    )
    assert config["experiment"]["probe_positives_per_class"] == 2
    assert config["experiment"]["dataset"] == "landcoverai"
    assert config["experiment"]["source_num_classes"] == 5
    assert config["experiment"]["target_classes"] == ["building", "road", "water"]
    assert config["experiment"]["seeds"] == [42, 123, 456]
    assert config["experiment"]["allow_synthetic_fallback"] is False
    assert config["text_encoder"]["model_id"] == "google/siglip-base-patch16-224"
    assert config["text_encoder"]["cache_dir"] is None
    assert config["prithvi"]["use_checkpoint_position_embeddings"] is True
    assert config["methods"] == [
        "siglip_film_dense_similarity_houlsby",
        "no_text_three_binary_heads_houlsby",
    ]
    assert config["model"] == {"condition_dim": 256, "decoder_dim": 128}
    assert config["training"] == {
        "lr": 0.001,
        "lr_peft": 0.0001,
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "positive_weight_clip": [1.0, 20.0],
    }
    assert config["evaluation"] == {
        "threshold": 0.5,
        "bootstrap_iterations": 1000,
        "preview_count": 12,
    }


def test_runner_build_text_encoder_forwards_explicit_cache(monkeypatch, tmp_path):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner
    import geoadapter.models.text_encoder as text_encoder_module

    calls = []

    class FakeSiglipTextEncoder:
        def __init__(
            self,
            model_id,
            *,
            revision=None,
            cache_dir=None,
            local_files_only=False,
        ):
            calls.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "cache_dir": cache_dir,
                    "local_files_only": local_files_only,
                }
            )

    monkeypatch.setattr(
        text_encoder_module, "SiglipTextEncoder", FakeSiglipTextEncoder
    )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["text_encoder"].update(
        {
            "model_id": "local/test-siglip",
            "revision": "seed42-recovery",
            "cache_dir": tmp_path / "huggingface-cache",
            "local_files_only": True,
        }
    )

    encoder = runner.build_text_encoder(config)

    assert isinstance(encoder, FakeSiglipTextEncoder)
    assert calls == [config["text_encoder"]]


def test_completed_keys_are_method_seed_pairs():
    rows = [{"method": "prompt", "seed": 42}, {"method": "baseline", "seed": 42}]
    assert completed_keys(rows) == {("prompt", 42), ("baseline", 42)}


def test_positive_weights_are_class_specific_and_clipped():
    masks = [
        torch.tensor([[1, 0], [3, 4]]),
        torch.tensor([[0, 0], [0, 0]]),
    ]
    weights = estimate_positive_weights(masks, clip=(1.0, 20.0))
    assert set(weights) == {"building", "road", "water"}
    assert all(1.0 <= value <= 20.0 for value in weights.values())
    assert weights == {"building": 7.0, "water": 7.0, "road": 7.0}


def test_checkpoint_metadata_hashes_external_contracts(tmp_path):
    prithvi = tmp_path / "prithvi.pt"
    prompts = tmp_path / "prompts.yaml"
    prithvi.write_bytes(b"weights")
    prompts.write_text("schema: test\n", encoding="utf-8")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["prithvi"]["checkpoint"] = str(prithvi)
    config["experiment"]["prompt_config"] = str(prompts)
    cache_dir = str(tmp_path / "huggingface-cache")
    config["text_encoder"]["cache_dir"] = cache_dir

    metadata = checkpoint_metadata(config, "prompt", 42)

    assert sha256_file(prithvi) == sha256_file(prithvi)
    assert metadata["schema"] == "paper12.geovlm_prompt_checkpoint.v2"
    assert metadata["training_contract"] == "paper12.geovlm_prompt_training.v2"
    assert metadata["target_pool_policy"] == "supported_target_present_only"
    assert metadata["empty_target_cap"] == 0.25
    assert metadata["probe_positives_per_class"] == 2
    assert metadata["best_checkpoint_policy"] == (
        "finite_nonconstant_prompt_change_loss_v1"
    )
    assert metadata["positive_weight_policy"] == (
        "full_source_training_split_v1"
    )
    assert metadata["prithvi_sha256"] == sha256_file(prithvi)
    assert metadata["prompt_config_sha256"] == sha256_file(prompts)
    assert metadata["class_mapping"] == {"building": 1, "water": 3, "road": 4}
    assert metadata["image_normalization"] == "rgb_float32_divide_255"
    assert "cache_dir" not in metadata
    assert cache_dir not in json.dumps(metadata, sort_keys=True)

    validate_checkpoint_metadata(metadata, metadata)
    for field in (
        "training_contract",
        "target_pool_policy",
        "empty_target_cap",
        "probe_positives_per_class",
        "best_checkpoint_policy",
        "positive_weight_policy",
    ):
        invalid = dict(metadata)
        invalid[field] = "mismatch"
        with pytest.raises(ValueError, match=field):
            validate_checkpoint_metadata(invalid, metadata)


def test_seed42_smoke_checks_report_failed_requirements():
    rows = [
        {
            "method": "siglip_film_dense_similarity_houlsby",
            "seed": 42,
            "class_name": class_name,
            "loss_history": [2.0, 1.0],
            "prediction_nonconstant": class_name != "road",
            "prompt_probability_change_by_sample": [0.02],
            "checkpoint_reproduced": True,
        }
        for class_name in ("building", "road", "water")
    ]

    smoke = seed42_smoke_checks(rows)

    assert smoke["passed"] is False
    assert smoke["checks"]["finite_decreasing_loss"] is True
    assert smoke["checks"]["nonconstant_predictions"] is False
    assert smoke["checks"]["prompt_dependent_probability_maps"] is True
    assert smoke["checks"]["checkpoint_reproduced"] is True
    assert smoke["failed_checks"] == ["nonconstant_predictions"]


def test_build_trainer_uses_configured_loss_coefficients():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["training"]["bce_weight"] = 0.25
    config["training"]["dice_weight"] = 1.75

    trainer = build_trainer(_TinyConditionalModel(), config, "cpu")

    assert trainer.criterion.bce_weight == 0.25
    assert trainer.criterion.dice_weight == 1.75


def test_train_epoch_uses_scheduled_conditions_and_reports_actual_stats():
    class _RecordingTrainer:
        device = "cpu"

        def __init__(self):
            self.conditions = None
            self.scheduler_steps = 0
            self.scheduler = type(
                "Scheduler",
                (),
                {"step": lambda scheduler: setattr(
                    self, "scheduler_steps", self.scheduler_steps + 1
                )},
            )()

        def train_step(self, _images, conditions, _targets, _positive_weights):
            self.conditions = conditions
            return 1.0

    trainer = _RecordingTrainer()
    images = torch.zeros(3, 3, 8, 8)
    masks = torch.stack(
        (
            torch.ones(8, 8, dtype=torch.long),
            torch.zeros(8, 8, dtype=torch.long),
            torch.full((8, 8), 4, dtype=torch.long),
        )
    )
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )

    stats = _train_one_epoch(
        trainer,
        [(images, masks, ("building", "water", "road"))],
        prompt_config,
        {"building": 1.0, "road": 1.0, "water": 1.0},
        42,
        BASELINE_METHOD,
    )

    assert trainer.conditions.tolist() == [1, 3, 4]
    assert trainer.scheduler_steps == 1
    assert stats == {
        "loss": 1.0,
        "sample_count": 3,
        "empty_target_count": 1,
        "prompt_counts": {"building": 1, "water": 1, "road": 1},
        "nonempty_prompt_counts": {"building": 1, "water": 0, "road": 1},
    }


def test_train_epoch_weights_loss_by_actual_batch_size():
    class _RecordingTrainer:
        device = "cpu"

        def __init__(self):
            self.losses = iter((1.0, 4.0))
            self.scheduler = type("Scheduler", (), {"step": lambda _self: None})()

        def train_step(self, _images, _conditions, _targets, _positive_weights):
            return next(self.losses)

    images = torch.zeros(3, 3, 8, 8)
    masks = torch.stack(
        (
            torch.ones(8, 8, dtype=torch.long),
            torch.full((8, 8), 3, dtype=torch.long),
            torch.full((8, 8), 4, dtype=torch.long),
        )
    )
    loader = [
        (images[:2], masks[:2], ("building", "water")),
        (images[2:], masks[2:], ("road",)),
    ]
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )

    stats = _train_one_epoch(
        _RecordingTrainer(),
        loader,
        prompt_config,
        {"building": 1.0, "road": 1.0, "water": 1.0},
        42,
        BASELINE_METHOD,
    )

    assert stats["sample_count"] == 3
    assert stats["loss"] == pytest.approx(2.0)


class _TinyPromptDataset(Dataset):
    def __init__(self):
        self.images = []
        self.masks = []
        for _ in range(3):
            for class_id, location in (
                (1, (0, 0)),
                (3, (0, 4)),
                (4, (4, 0)),
            ):
                mask = torch.zeros(8, 8, dtype=torch.long)
                row, col = location
                mask[row : row + 4, col : col + 4] = class_id
                image = torch.zeros(3, 8, 8)
                image[0] = mask.eq(1)
                image[1] = mask.eq(3)
                image[2] = mask.eq(4)
                self.images.append(image)
                self.masks.append(mask)
        self.images.append(torch.zeros(3, 8, 8))
        self.masks.append(torch.full((8, 8), 2, dtype=torch.long))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]


class _TinyConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.condition_bias = nn.Parameter(torch.tensor([0.0, 0.25, 0.5]))
        with torch.no_grad():
            self.conv.weight.fill_(4.0)
            self.conv.bias.fill_(-2.0)

    @staticmethod
    def _condition_indices(conditions):
        if torch.is_tensor(conditions):
            mapping = {1: 0, 3: 1, 4: 2}
            return [mapping[int(value)] for value in conditions]
        indices = []
        for prompt in conditions:
            if "building" in prompt or "structure" in prompt or "roof" in prompt:
                indices.append(0)
            elif "water" in prompt or "aquatic" in prompt or "lake" in prompt:
                indices.append(1)
            else:
                indices.append(2)
        return indices

    def forward(self, images, conditions):
        logits = self.conv(images)[:, 0]
        indices = torch.tensor(self._condition_indices(conditions), device=images.device)
        return logits + self.condition_bias[indices, None, None]


class _NonFiniteConditionalModel(_TinyConditionalModel):
    def forward(self, images, conditions):
        return super().forward(images, conditions) * float("nan")


class _SpatiallyConstantDataset(Dataset):
    def __init__(self):
        self.samples = []
        for index, class_id in enumerate((1, 1, 3, 3, 4, 4), start=1):
            image = torch.full((3, 8, 8), float(index))
            mask = torch.full((8, 8), class_id, dtype=torch.long)
            self.samples.append((image, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _SpatiallyConstantBaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.condition_dtypes = []

    def forward(self, images, conditions):
        if not torch.is_tensor(conditions):
            raise TypeError("baseline probe conditions must be tensors")
        self.condition_dtypes.append(conditions.dtype)
        levels = images[:, 0].mean(dim=(1, 2)) * self.scale
        return levels[:, None, None].expand(-1, images.shape[-2], images.shape[-1])


def test_training_probe_is_finite_nonconstant_and_prompt_dependent():
    class RecordingDataset(_TinyPromptDataset):
        def __init__(self):
            super().__init__()
            self.accessed_indices = []

        def __getitem__(self, index):
            self.accessed_indices.append(index)
            return super().__getitem__(index)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )
    train = RecordingDataset()
    trainer = build_trainer(_TinyConditionalModel(), config, "cpu")
    with torch.no_grad():
        trainer.model.condition_bias.copy_(torch.tensor([0.0, 1.0, 2.0]))

    prompt_condition = _probe_condition(PROMPT_METHOD, "building", prompt_config)
    baseline_condition = _probe_condition(BASELINE_METHOD, "building", prompt_config)
    probe = _evaluate_probe(
        trainer,
        train,
        {"building": [0], "water": [1], "road": [2]},
        prompt_config,
        {"building": 1.0, "water": 1.0, "road": 1.0},
        PROMPT_METHOD,
    )

    assert prompt_condition == [prompt_config.classes["building"].training[0]]
    assert prompt_condition[0] not in prompt_config.classes["building"].held_out
    assert baseline_condition.dtype == torch.long
    assert baseline_condition.device.type == "cpu"
    assert baseline_condition.tolist() == [1]
    assert train.accessed_indices == [0, 1, 2]
    assert probe["finite"] is True
    assert probe["mean_loss"] > 0.0
    assert probe["nonconstant_class_count"] == 3
    assert probe["prompt_map_changed_class_count"] == 3
    assert probe["mean_prompt_probability_change"] > 0.0
    assert list(probe["classes"]) == ["building", "water", "road"]
    assert json.loads(json.dumps(probe, allow_nan=False)) == probe
    rank = probe_rank(probe)
    assert rank[:3] == (1, 3, 3)
    assert rank[3] > 0.0
    assert rank[4] == -probe["mean_loss"]


def test_probe_rank_uses_lexicographic_checkpoint_policy():
    base = {
        "finite": True,
        "nonconstant_class_count": 3,
        "prompt_map_changed_class_count": 3,
        "mean_prompt_probability_change": 0.2,
        "mean_loss": 1.0,
    }

    base_rank = probe_rank(base)

    assert probe_rank({**base, "mean_prompt_probability_change": 0.3, "mean_loss": 2.0}) > base_rank
    assert probe_rank({**base, "mean_loss": 0.5}) > base_rank
    assert probe_rank(
        {
            "finite": False,
            "nonconstant_class_count": 999,
            "prompt_map_changed_class_count": 999,
            "mean_prompt_probability_change": float("nan"),
            "mean_loss": float("nan"),
        }
    ) == (0, 0, 0, 0.0, 0.0)


def test_nonfinite_probe_is_strict_json_and_gets_finite_rank_sentinel():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )
    trainer = build_trainer(_NonFiniteConditionalModel(), config, "cpu")

    probe = _evaluate_probe(
        trainer,
        _TinyPromptDataset(),
        {"building": [0], "water": [1], "road": [2]},
        prompt_config,
        {"building": 1.0, "water": 1.0, "road": 1.0},
        PROMPT_METHOD,
    )

    assert probe["finite"] is False
    assert probe["mean_loss"] is None
    assert probe["mean_prompt_probability_change"] is None
    assert all(
        diagnostics == {
            "prediction_range": None,
            "prediction_nonconstant": False,
            "mean_prompt_probability_change": None,
            "prompt_map_changed": False,
        }
        for diagnostics in probe["classes"].values()
    )
    json.dumps(probe, allow_nan=False)
    assert probe_rank(probe) == (0, 0, 0, 0.0, 0.0)


def test_atomic_json_rejects_nonfinite_payload_without_writing_target(tmp_path):
    output = tmp_path / "probe.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        _atomic_json(output, {"mean_loss": float("nan")})

    assert not output.exists()


def test_baseline_probe_uses_spatial_range_and_restores_model_mode():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )
    trainer = build_trainer(_SpatiallyConstantBaselineModel(), config, "cpu")
    probe_indices = {"building": [0, 1], "water": [2, 3], "road": [4, 5]}
    weights = {"building": 1.0, "water": 1.0, "road": 1.0}
    dataset = _SpatiallyConstantDataset()

    trainer.model.train()
    training_probe = _evaluate_probe(
        trainer,
        dataset,
        probe_indices,
        prompt_config,
        weights,
        BASELINE_METHOD,
    )
    assert trainer.model.training is True

    trainer.model.eval()
    evaluation_probe = _evaluate_probe(
        trainer,
        dataset,
        probe_indices,
        prompt_config,
        weights,
        BASELINE_METHOD,
    )

    assert trainer.model.training is False
    assert training_probe == evaluation_probe
    assert all(dtype == torch.long for dtype in trainer.model.condition_dtypes)
    assert training_probe["nonconstant_class_count"] == 0
    assert all(
        diagnostics["prediction_range"] == 0.0
        and diagnostics["prediction_nonconstant"] is False
        for diagnostics in training_probe["classes"].values()
    )


@pytest.mark.parametrize("method", [PROMPT_METHOD, BASELINE_METHOD])
def test_checkpoint_reproduction_checks_all_target_conditions(method, tmp_path):
    class _RecordingConditionalModel(_TinyConditionalModel):
        def __init__(self):
            super().__init__()
            self.condition_calls = []

        def forward(self, images, conditions):
            self.condition_calls.append(tuple(self._condition_indices(conditions)))
            return super().forward(images, conditions)

    class _RecordingValidation(_TinyPromptDataset):
        def __init__(self):
            super().__init__()
            self.accessed_indices = []

        def __getitem__(self, index):
            self.accessed_indices.append(index)
            return super().__getitem__(index)

    config = _runner_test_config(tmp_path, epochs=1)
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )
    active_model = _RecordingConditionalModel()
    active_trainer = build_trainer(active_model, config, "cpu")
    checkpoint_model = _TinyConditionalModel()
    checkpoint_model.load_state_dict(active_model.state_dict())
    with torch.no_grad():
        checkpoint_model.condition_bias[1].add_(1.0)
    checkpoint_trainer = build_trainer(checkpoint_model, config, "cpu")
    checkpoint_path = tmp_path / f"{method}.pt"
    torch.save(
        checkpoint_trainer.state_dict(epoch=1, metadata={}),
        checkpoint_path,
    )
    clones = []

    def model_builder(_config, _method, device):
        clone = _RecordingConditionalModel().to(device)
        clones.append(clone)
        return clone

    validation = _RecordingValidation()
    reproduced = _checkpoint_reproduces(
        config,
        method,
        123,
        checkpoint_path,
        active_trainer,
        model_builder,
        validation,
        prompt_config,
        "cpu",
    )

    expected_calls = [(0,), (1,), (2,)]
    assert validation.accessed_indices == [0]
    assert active_model.condition_calls == expected_calls
    assert clones[0].condition_calls == expected_calls
    assert reproduced is False


@pytest.mark.parametrize(
    ("probe_indices", "message"),
    [
        (
            {"building": [0], "water": [1]},
            "probe_indices_by_class must contain exactly",
        ),
        (
            {"building": [0], "water": [], "road": [2]},
            "probe indices for water must be non-empty",
        ),
    ],
)
def test_probe_requires_all_nonempty_class_indices(probe_indices, message):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )
    trainer = build_trainer(_TinyConditionalModel(), config, "cpu")

    with pytest.raises(ValueError, match=message):
        _evaluate_probe(
            trainer,
            _TinyPromptDataset(),
            probe_indices,
            prompt_config,
            {"building": 1.0, "water": 1.0, "road": 1.0},
            PROMPT_METHOD,
        )


def _runner_test_config(tmp_path, *, epochs=2, probe_positives_per_class=1):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prithvi = tmp_path / "prithvi.pt"
    prithvi.write_bytes(b"tiny-prithvi")
    config["prithvi"]["checkpoint"] = str(prithvi)
    config["experiment"]["prompt_config"] = str(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml").resolve()
    )
    config["experiment"]["epochs"] = epochs
    config["experiment"]["batch_size"] = 2
    config["experiment"]["probe_positives_per_class"] = (
        probe_positives_per_class
    )
    config["evaluation"]["preview_count"] = 0
    return config


def _finite_probe(change, loss):
    classes = {
        class_name: {
            "prediction_range": 1.0,
            "prediction_nonconstant": True,
            "mean_prompt_probability_change": float(change),
            "prompt_map_changed": True,
        }
        for class_name in ("building", "water", "road")
    }
    return {
        "finite": True,
        "mean_loss": float(loss),
        "nonconstant_class_count": 3,
        "prompt_map_changed_class_count": 3,
        "mean_prompt_probability_change": float(change),
        "classes": classes,
    }


def _epoch_stats(loss, *, sample_count=6):
    per_class = sample_count // 3
    return {
        "loss": float(loss),
        "sample_count": sample_count,
        "empty_target_count": 0,
        "prompt_counts": {
            "building": per_class,
            "water": per_class,
            "road": per_class,
        },
        "nonempty_prompt_counts": {
            "building": per_class,
            "water": per_class,
            "road": per_class,
        },
    }


def _fake_evaluation_rows(method, seed):
    return [
        {"method": method, "seed": seed, "class_name": class_name}
        for class_name in ("building", "water", "road")
    ]


def _resume_state(
    config,
    model_builder,
    *,
    method,
    seed,
    epoch,
    losses,
    probes,
    training_stats,
    probe_indices,
    probe_sha256,
    positive_weights,
    best_epoch,
):
    best_rank = list(probe_rank(probes[best_epoch - 1]))
    metadata = checkpoint_metadata(config, method, seed)
    metadata.update(
        {"best_epoch": best_epoch, "best_probe_rank": best_rank}
    )
    trainer = build_trainer(model_builder(config, method, "cpu"), config, "cpu")
    with torch.no_grad():
        trainer.model.condition_bias.fill_(float(epoch))
    state = trainer.state_dict(epoch=epoch, metadata=metadata)
    state.update(
        {
            "loss_history": list(losses),
            "probe_history": copy.deepcopy(probes),
            "training_stats": copy.deepcopy(training_stats),
            "probe_indices_by_class": copy.deepcopy(probe_indices),
            "probe_sha256": probe_sha256,
            "positive_weights": dict(positive_weights),
            "best_epoch": best_epoch,
            "best_probe_rank": best_rank,
        }
    )
    return state


def _two_epoch_resume_contract(config, dataset, model_builder, *, best_epoch):
    split = reserve_training_probe(
        scan_target_present_pool(dataset), seed=123, positives_per_class=1
    )
    probe_indices = {
        class_name: list(indices)
        for class_name, indices in split.probe_indices_by_class.items()
    }
    probes = [
        {
            **_finite_probe(1.0, 2.0),
            "epoch": 1,
            "training_loss": 2.0,
        },
        {
            **_finite_probe(0.5, 3.0),
            "epoch": 2,
            "training_loss": 3.0,
        },
    ]
    positive_weights = estimate_positive_weights(
        (dataset[index][1] for index in range(len(dataset))),
        clip=tuple(config["training"]["positive_weight_clip"]),
    )
    last = _resume_state(
        config,
        model_builder,
        method=BASELINE_METHOD,
        seed=123,
        epoch=2,
        losses=[2.0, 3.0],
        probes=probes,
        training_stats={
            key: value
            for key, value in _epoch_stats(3.0, sample_count=12).items()
            if key != "loss"
        },
        probe_indices=probe_indices,
        probe_sha256=split.probe_sha256,
        positive_weights=positive_weights,
        best_epoch=best_epoch,
    )
    return (
        last,
        probes,
        probe_indices,
        split.probe_sha256,
        positive_weights,
    )


def test_runner_rejects_full_source_positive_weight_drift_before_writing(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path)
    dataset = _TinyPromptDataset()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    (
        last,
        probes,
        probe_indices,
        probe_sha256,
        positive_weights,
    ) = _two_epoch_resume_contract(
        config, dataset, model_builder, best_epoch=1
    )
    best = _resume_state(
        config,
        model_builder,
        method=BASELINE_METHOD,
        seed=123,
        epoch=1,
        losses=[2.0],
        probes=probes[:1],
        training_stats={
            key: value
            for key, value in _epoch_stats(2.0).items()
            if key != "loss"
        },
        probe_indices=probe_indices,
        probe_sha256=probe_sha256,
        positive_weights=positive_weights,
        best_epoch=1,
    )
    base = checkpoint_dir / f"{BASELINE_METHOD}__seed123"
    last_path = base.with_suffix(".last.pt")
    best_path = base.with_suffix(".best.pt")
    torch.save(last, last_path)
    torch.save(best, best_path)
    before = {path: path.read_bytes() for path in (last_path, best_path)}

    dataset.masks[0].fill_(1)
    monkeypatch.setattr(
        runner,
        "_evaluate_method",
        lambda _trainer, _validation, _prompts, method, seed, *_args: (
            _fake_evaluation_rows(method, seed)
        ),
    )

    with pytest.raises(
        ValueError,
        match="positive weights mismatch.*full source",
    ):
        runner._run_pair(
            config,
            BASELINE_METHOD,
            123,
            dataset,
            dataset,
            checkpoint_dir,
            tmp_path / "previews",
            "cpu",
            model_builder,
        )

    assert {path: path.read_bytes() for path in before} == before


def test_runner_rejects_checkpoint_training_sample_count_mismatch(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path)
    dataset = _TinyPromptDataset()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    (
        last,
        probes,
        probe_indices,
        probe_sha256,
        positive_weights,
    ) = _two_epoch_resume_contract(
        config, dataset, model_builder, best_epoch=1
    )
    last["training_stats"] = {
        key: value
        for key, value in _epoch_stats(3.0, sample_count=9).items()
        if key != "loss"
    }
    best = _resume_state(
        config,
        model_builder,
        method=BASELINE_METHOD,
        seed=123,
        epoch=1,
        losses=[2.0],
        probes=probes[:1],
        training_stats={
            key: value
            for key, value in _epoch_stats(2.0).items()
            if key != "loss"
        },
        probe_indices=probe_indices,
        probe_sha256=probe_sha256,
        positive_weights=positive_weights,
        best_epoch=1,
    )
    base = checkpoint_dir / f"{BASELINE_METHOD}__seed123"
    last_path = base.with_suffix(".last.pt")
    best_path = base.with_suffix(".best.pt")
    torch.save(last, last_path)
    torch.save(best, best_path)
    before = {path: path.read_bytes() for path in (last_path, best_path)}
    monkeypatch.setattr(
        runner,
        "_evaluate_method",
        lambda _trainer, _validation, _prompts, method, seed, *_args: (
            _fake_evaluation_rows(method, seed)
        ),
    )

    with pytest.raises(
        ValueError,
        match="checkpoint training sample count mismatch.*checkpoint epoch",
    ):
        runner._run_pair(
            config,
            BASELINE_METHOD,
            123,
            dataset,
            dataset,
            checkpoint_dir,
            tmp_path / "previews",
            "cpu",
            model_builder,
        )

    assert {path: path.read_bytes() for path in before} == before


def test_runner_rejects_new_epoch_training_sample_count_mismatch(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path, epochs=1)
    dataset = _TinyPromptDataset()
    checkpoint_dir = tmp_path / "checkpoints"

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    monkeypatch.setattr(
        runner,
        "_train_one_epoch",
        lambda *_args, **_kwargs: _epoch_stats(1.0, sample_count=3),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_probe",
        lambda *_args, **_kwargs: _finite_probe(1.0, 1.0),
    )

    with pytest.raises(
        ValueError,
        match="training epoch sample count mismatch",
    ):
        runner._run_pair(
            config,
            BASELINE_METHOD,
            123,
            dataset,
            dataset,
            checkpoint_dir,
            tmp_path / "previews",
            "cpu",
            model_builder,
        )

    assert not list(checkpoint_dir.glob("*.pt"))


def test_runner_recomputes_best_selection_from_last_probe_history(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path)
    dataset = _TinyPromptDataset()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    last, _, _, _, _ = _two_epoch_resume_contract(
        config, dataset, model_builder, best_epoch=2
    )
    base = checkpoint_dir / f"{BASELINE_METHOD}__seed123"
    last_path = base.with_suffix(".last.pt")
    best_path = base.with_suffix(".best.pt")
    torch.save(last, last_path)
    torch.save(copy.deepcopy(last), best_path)
    before = {path: path.read_bytes() for path in (last_path, best_path)}
    monkeypatch.setattr(
        runner,
        "_evaluate_method",
        lambda _trainer, _validation, _prompts, method, seed, *_args: (
            _fake_evaluation_rows(method, seed)
        ),
    )

    with pytest.raises(ValueError, match="best selection mismatch"):
        runner._run_pair(
            config,
            BASELINE_METHOD,
            123,
            dataset,
            dataset,
            checkpoint_dir,
            tmp_path / "previews",
            "cpu",
            model_builder,
        )

    assert {path: path.read_bytes() for path in before} == before


def test_runner_rejects_last_metadata_best_selection_mismatch(tmp_path):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path)
    dataset = _TinyPromptDataset()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    last, probes, _, _, _ = _two_epoch_resume_contract(
        config, dataset, model_builder, best_epoch=1
    )
    last["metadata"].update(
        {
            "best_epoch": 2,
            "best_probe_rank": list(probe_rank(probes[1])),
        }
    )
    base = checkpoint_dir / f"{BASELINE_METHOD}__seed123"
    last_path = base.with_suffix(".last.pt")
    best_path = base.with_suffix(".best.pt")
    torch.save(last, last_path)
    torch.save(copy.deepcopy(last), best_path)
    before = {path: path.read_bytes() for path in (last_path, best_path)}

    with pytest.raises(ValueError, match="best selection mismatch"):
        runner._run_pair(
            config,
            BASELINE_METHOD,
            123,
            dataset,
            dataset,
            checkpoint_dir,
            tmp_path / "previews",
            "cpu",
            model_builder,
        )

    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("corrupted_field", ["loss_history", "probe_history"])
def test_runner_rejects_best_history_that_is_not_last_prefix(
    monkeypatch, tmp_path, corrupted_field
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path)
    dataset = _TinyPromptDataset()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    (
        last,
        probes,
        probe_indices,
        probe_sha256,
        positive_weights,
    ) = _two_epoch_resume_contract(
        config, dataset, model_builder, best_epoch=1
    )
    best = _resume_state(
        config,
        model_builder,
        method=BASELINE_METHOD,
        seed=123,
        epoch=1,
        losses=[2.0],
        probes=probes[:1],
        training_stats={
            key: value
            for key, value in _epoch_stats(2.0).items()
            if key != "loss"
        },
        probe_indices=probe_indices,
        probe_sha256=probe_sha256,
        positive_weights=positive_weights,
        best_epoch=1,
    )
    if corrupted_field == "loss_history":
        best["loss_history"] = [9.0]
        best["probe_history"][0]["training_loss"] = 9.0
    else:
        best["probe_history"][0]["classes"]["building"][
            "prediction_range"
        ] = 9.0

    base = checkpoint_dir / f"{BASELINE_METHOD}__seed123"
    last_path = base.with_suffix(".last.pt")
    best_path = base.with_suffix(".best.pt")
    torch.save(last, last_path)
    torch.save(best, best_path)
    before = {path: path.read_bytes() for path in (last_path, best_path)}
    monkeypatch.setattr(
        runner,
        "_evaluate_method",
        lambda _trainer, _validation, _prompts, method, seed, *_args: (
            _fake_evaluation_rows(method, seed)
        ),
    )

    with pytest.raises(
        ValueError,
        match=f"best checkpoint history mismatch: {corrupted_field}",
    ):
        runner._run_pair(
            config,
            BASELINE_METHOD,
            123,
            dataset,
            dataset,
            checkpoint_dir,
            tmp_path / "previews",
            "cpu",
            model_builder,
        )

    assert {path: path.read_bytes() for path in before} == before


def test_runner_rejects_legacy_and_lone_best_checkpoints_before_building(tmp_path):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path)
    dataset = _TinyPromptDataset()

    def forbidden_model_builder(*_args):
        raise AssertionError("model must not be built for invalid checkpoint layout")

    legacy_dir = tmp_path / "legacy-checkpoints"
    legacy_dir.mkdir()
    legacy_path = legacy_dir / f"{BASELINE_METHOD}__seed123.pt"
    legacy_path.write_bytes(b"legacy")
    with pytest.raises(ValueError, match="archive it before recovery"):
        runner._run_pair(
            config,
            BASELINE_METHOD,
            123,
            dataset,
            dataset,
            legacy_dir,
            tmp_path / "previews",
            "cpu",
            forbidden_model_builder,
        )
    assert list(legacy_dir.iterdir()) == [legacy_path]

    lone_best_dir = tmp_path / "lone-best-checkpoints"
    lone_best_dir.mkdir()
    best_path = lone_best_dir / f"{BASELINE_METHOD}__seed123.best.pt"
    best_path.write_bytes(b"best")
    with pytest.raises(ValueError, match="best checkpoint exists without last"):
        runner._run_pair(
            config,
            BASELINE_METHOD,
            123,
            dataset,
            dataset,
            lone_best_dir,
            tmp_path / "previews",
            "cpu",
            forbidden_model_builder,
        )
    assert list(lone_best_dir.iterdir()) == [best_path]


def test_runner_evaluates_best_probe_checkpoint_when_later_epoch_degrades(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    dataset = _TinyPromptDataset()
    train_calls = []
    evaluation_calls = []

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    def fake_train_one_epoch(
        trainer, loader, _prompt_config, _weights, epoch_seed, _method
    ):
        epoch = len(train_calls) + 1
        train_calls.append((epoch_seed, len(loader.dataset)))
        with torch.no_grad():
            trainer.model.condition_bias.fill_(float(epoch))
        return _epoch_stats(epoch)

    def fake_probe(trainer, *_args):
        bias = float(trainer.model.condition_bias[0].detach())
        return _finite_probe(1.0 if bias == 1.0 else 0.5, bias)

    def fake_evaluate(trainer, validation, _prompt_config, method, seed, *_args):
        evaluation_calls.append(
            (len(validation), trainer.model.condition_bias.detach().tolist())
        )
        return _fake_evaluation_rows(method, seed)

    monkeypatch.setattr(runner, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(runner, "_evaluate_probe", fake_probe)
    monkeypatch.setattr(runner, "_evaluate_method", fake_evaluate)

    rows = runner._run_pair(
        config,
        BASELINE_METHOD,
        123,
        dataset,
        dataset,
        checkpoint_dir,
        tmp_path / "previews",
        "cpu",
        model_builder,
    )

    base = checkpoint_dir / f"{BASELINE_METHOD}__seed123"
    last = torch.load(
        base.with_suffix(".last.pt"), map_location="cpu", weights_only=False
    )
    best = torch.load(
        base.with_suffix(".best.pt"), map_location="cpu", weights_only=False
    )
    expected_rank = list(probe_rank(_finite_probe(1.0, 1.0)))
    expected_weights = estimate_positive_weights(
        (dataset[index][1] for index in range(len(dataset))),
        clip=tuple(config["training"]["positive_weight_clip"]),
    )
    assert train_calls == [(123, 6), (124, 6)]
    assert last["epoch"] == 2
    assert last["trainable_model"]["condition_bias"].tolist() == [2.0] * 3
    assert best["epoch"] == 1
    assert best["trainable_model"]["condition_bias"].tolist() == [1.0] * 3
    assert evaluation_calls == [(10, [1.0, 1.0, 1.0])]
    assert last["best_epoch"] == best["best_epoch"] == 1
    assert last["best_probe_rank"] == best["best_probe_rank"] == expected_rank
    assert last["metadata"]["best_epoch"] == best["metadata"]["best_epoch"] == 1
    assert last["positive_weights"] == best["positive_weights"] == expected_weights
    assert (
        last["metadata"]["positive_weight_policy"]
        == best["metadata"]["positive_weight_policy"]
        == "full_source_training_split_v1"
    )
    assert (
        last["metadata"]["best_probe_rank"]
        == best["metadata"]["best_probe_rank"]
        == expected_rank
    )
    assert all(row["checkpoint_reproduced"] is True for row in rows)
    assert all(row["best_epoch"] == 1 for row in rows)
    assert all(row["full_loss_history"] == [1.0, 2.0] for row in rows)
    assert all(row["loss_history"] == [1.0] for row in rows)


def test_runner_rows_have_independent_best_probe_diagnostics(monkeypatch, tmp_path):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path, epochs=1)
    dataset = _TinyPromptDataset()

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    monkeypatch.setattr(
        runner,
        "_train_one_epoch",
        lambda *_args, **_kwargs: _epoch_stats(1.0),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_probe",
        lambda *_args, **_kwargs: _finite_probe(1.0, 1.0),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_method",
        lambda _trainer, _validation, _prompts, method, seed, *_args: (
            _fake_evaluation_rows(method, seed)
        ),
    )
    monkeypatch.setattr(runner, "_checkpoint_reproduces", lambda *_args: True)

    rows = runner._run_pair(
        config,
        BASELINE_METHOD,
        123,
        dataset,
        dataset,
        tmp_path / "checkpoints",
        tmp_path / "previews",
        "cpu",
        model_builder,
    )

    rows[0]["best_probe"]["classes"]["building"]["prediction_range"] = 99.0

    assert len(rows) == 3
    assert rows[1]["best_probe"]["classes"]["building"]["prediction_range"] == 1.0
    assert rows[2]["best_probe"]["classes"]["building"]["prediction_range"] == 1.0


def test_runner_resumes_last_checkpoint_without_replacing_better_epoch_one(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    dataset = _TinyPromptDataset()
    pool = scan_target_present_pool(dataset)
    split = reserve_training_probe(pool, seed=123, positives_per_class=1)
    probe_indices = {
        class_name: list(indices)
        for class_name, indices in split.probe_indices_by_class.items()
    }
    first_probe = {
        **_finite_probe(1.0, 2.0),
        "epoch": 1,
        "training_loss": 2.0,
    }
    best_rank = list(probe_rank(first_probe))
    metadata = checkpoint_metadata(config, BASELINE_METHOD, 123)
    metadata.update({"best_epoch": 1, "best_probe_rank": best_rank})

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    trainer = build_trainer(
        model_builder(config, BASELINE_METHOD, "cpu"), config, "cpu"
    )
    positive_weights = estimate_positive_weights(
        (dataset[index][1] for index in range(len(dataset))),
        clip=tuple(config["training"]["positive_weight_clip"]),
    )
    with torch.no_grad():
        trainer.model.condition_bias.fill_(1.0)
    state = trainer.state_dict(epoch=1, metadata=metadata)
    state.update(
        {
            "loss_history": [2.0],
            "probe_history": [first_probe],
            "training_stats": {
                key: value
                for key, value in _epoch_stats(2.0).items()
                if key != "loss"
            },
            "probe_indices_by_class": probe_indices,
            "probe_sha256": split.probe_sha256,
            "positive_weights": positive_weights,
            "best_epoch": 1,
            "best_probe_rank": best_rank,
        }
    )
    base = checkpoint_dir / f"{BASELINE_METHOD}__seed123"
    torch.save(state, base.with_suffix(".last.pt"))
    torch.save(state, base.with_suffix(".best.pt"))
    train_calls = []

    def fake_train_one_epoch(
        resumed_trainer, loader, _prompt_config, _weights, epoch_seed, _method
    ):
        train_calls.append((epoch_seed, len(loader.dataset)))
        with torch.no_grad():
            resumed_trainer.model.condition_bias.fill_(2.0)
        return _epoch_stats(3.0)

    def fake_probe(_trainer, *_args):
        return _finite_probe(0.5, 3.0)

    def fake_evaluate(_trainer, _validation, _prompt_config, method, seed, *_args):
        return _fake_evaluation_rows(method, seed)

    monkeypatch.setattr(runner, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(runner, "_evaluate_probe", fake_probe)
    monkeypatch.setattr(runner, "_evaluate_method", fake_evaluate)

    rows = runner._run_pair(
        config,
        BASELINE_METHOD,
        123,
        dataset,
        dataset,
        checkpoint_dir,
        tmp_path / "previews",
        "cpu",
        model_builder,
    )

    resumed = torch.load(
        base.with_suffix(".last.pt"), map_location="cpu", weights_only=False
    )
    selected = torch.load(
        base.with_suffix(".best.pt"), map_location="cpu", weights_only=False
    )
    assert train_calls == [(124, 6)]
    assert resumed["epoch"] == 2
    assert resumed["loss_history"] == [2.0, 3.0]
    assert resumed["best_epoch"] == 1
    assert resumed["training_stats"]["sample_count"] == 12
    assert selected["epoch"] == 1
    assert selected["loss_history"] == [2.0]
    assert selected["trainable_model"]["condition_bias"].tolist() == [1.0] * 3
    assert all(row["best_epoch"] == 1 for row in rows)
    assert all(row["full_loss_history"] == [2.0, 3.0] for row in rows)
    assert all(row["loss_history"] == [2.0] for row in rows)


def test_runner_appends_skips_and_reloads_with_injected_builders(
    monkeypatch, tmp_path
):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = _runner_test_config(
        tmp_path, epochs=2, probe_positives_per_class=2
    )
    config["experiment"]["seeds"] = [42]
    config["evaluation"]["preview_count"] = 4
    output = tmp_path / "rows.json"
    summary_output = tmp_path / "summary.json"
    checkpoint_dir = tmp_path / "checkpoints"
    preview_dir = tmp_path / "previews"
    build_calls = []
    train_calls = []
    probe_calls = []
    real_train_one_epoch = runner._train_one_epoch
    real_evaluate_probe = runner._evaluate_probe

    def dataset_builder(_config):
        dataset = _TinyPromptDataset()
        return dataset, dataset

    def model_builder(_config, method, device):
        build_calls.append(method)
        return _TinyConditionalModel().to(device)

    def deterministic_train(*args, **kwargs):
        stats = real_train_one_epoch(*args, **kwargs)
        stats["loss"] = (2.0, 1.0)[len(train_calls) % 2]
        train_calls.append(stats["loss"])
        return stats

    def deterministic_probe(*args, **kwargs):
        probe = real_evaluate_probe(*args, **kwargs)
        probe["mean_prompt_probability_change"] = (0.1, 0.2)[
            len(probe_calls) % 2
        ]
        probe_calls.append(probe["mean_prompt_probability_change"])
        return probe

    monkeypatch.setattr(runner, "_train_one_epoch", deterministic_train)
    monkeypatch.setattr(runner, "_evaluate_probe", deterministic_probe)

    rows = run_experiment(
        config,
        output_path=output,
        summary_output_path=summary_output,
        checkpoint_dir=checkpoint_dir,
        preview_dir=preview_dir,
        stage="full",
        device="cpu",
        dataset_builder=dataset_builder,
        model_builder=model_builder,
    )

    assert len(rows) == 6
    assert len(list(checkpoint_dir.glob("*.pt"))) == 4
    assert all(row.get("synthetic_fallback") is not True for row in rows)
    prompt_rows = [row for row in rows if row["method"].startswith("siglip")]
    baseline_rows = [row for row in rows if row["method"].startswith("no_text")]
    assert all("correct_iou_by_sample" in row for row in prompt_rows)
    assert all("correct_iou_by_sample" not in row for row in baseline_rows)
    assert all(row["checkpoint_reproduced"] is True for row in rows)
    assert all(
        row["training_contract"] == "paper12.geovlm_prompt_training.v2"
        for row in rows
    )
    assert all(row["source_training_size"] == 10 for row in rows)
    assert all(row["target_present_pool_size"] == 9 for row in rows)
    assert all(row["excluded_no_target_count"] == 1 for row in rows)
    assert all(row["excluded_no_target_share"] == 0.1 for row in rows)
    assert all(
        set(row["probe_indices_by_class"]) == {"building", "water", "road"}
        for row in rows
    )
    assert all(len(row["probe_sha256"]) == 64 for row in rows)
    assert all(
        set(row["per_class_prompt_counts"]) == {"building", "water", "road"}
        for row in rows
    )
    assert all(
        set(row["per_class_nonempty_prompt_counts"])
        == {"building", "water", "road"}
        for row in rows
    )
    assert all(row["observed_training_sample_count"] > 0 for row in rows)
    assert all(row["observed_empty_target_count"] >= 0 for row in rows)
    assert all(row["observed_empty_target_share"] <= 0.25 for row in rows)
    assert all(len(row["full_loss_history"]) == 2 for row in rows)
    assert all(row["best_epoch"] == 2 for row in rows)
    assert all(len(row["loss_history"]) == row["best_epoch"] for row in rows)
    assert all(row["best_probe"]["epoch"] == 2 for row in rows)
    assert all(row["best_probe"]["training_loss"] == 1.0 for row in rows)
    assert all(len(row["best_probe_rank"]) == 5 for row in rows)
    preview_paths = [
        Path(path)
        for row in rows
        for path in row.get("preview_paths", [])
    ]
    assert len(preview_paths) == 4
    assert all(path.is_file() and path.suffix == ".png" for path in preview_paths)
    assert all("seed42" in path.name for path in preview_paths)
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["schema"] == "paper12.geovlm_prompt_results.v1"
    assert raw["seed42_smoke"]["passed"] is True
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["mvp_status"] == "incomplete"

    calls_after_first_run = len(build_calls)
    repeated = run_experiment(
        config,
        output_path=output,
        summary_output_path=summary_output,
        checkpoint_dir=checkpoint_dir,
        preview_dir=preview_dir,
        stage="full",
        device="cpu",
        dataset_builder=dataset_builder,
        model_builder=model_builder,
    )
    assert repeated == rows
    assert len(build_calls) == calls_after_first_run
