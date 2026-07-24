import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

from geoadapter.bench.run_geovlm_prompt_segmentation import (
    BASELINE_METHOD,
    _train_one_epoch,
    build_trainer,
    checkpoint_metadata,
    completed_keys,
    estimate_positive_weights,
    run_experiment,
    seed42_smoke_checks,
    sha256_file,
    validate_checkpoint_metadata,
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
    assert metadata["prithvi_sha256"] == sha256_file(prithvi)
    assert metadata["prompt_config_sha256"] == sha256_file(prompts)
    assert metadata["class_mapping"] == {"building": 1, "water": 3, "road": 4}
    assert metadata["image_normalization"] == "rgb_float32_divide_255"
    assert "cache_dir" not in metadata
    assert str(tmp_path) not in json.dumps(metadata)

    validate_checkpoint_metadata(metadata, metadata)
    for field in (
        "training_contract",
        "target_pool_policy",
        "empty_target_cap",
        "probe_positives_per_class",
        "best_checkpoint_policy",
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


def test_train_epoch_passes_configured_empty_target_cap():
    class _RecordingTrainer:
        device = "cpu"

        def __init__(self):
            self.conditions = None
            self.scheduler = type("Scheduler", (), {"step": lambda self: None})()

        def train_step(self, _images, conditions, _targets, _positive_weights):
            self.conditions = conditions
            return 1.0

    trainer = _RecordingTrainer()
    images = torch.zeros(4, 3, 8, 8)
    masks = torch.ones(4, 8, 8, dtype=torch.long)
    prompt_config = load_prompt_config(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml")
    )

    _train_one_epoch(
        trainer,
        [(images, masks)],
        prompt_config,
        {"building": 1.0, "road": 1.0, "water": 1.0},
        42,
        BASELINE_METHOD,
        empty_target_cap=0.0,
    )

    assert trainer.conditions.tolist() == [1, 1, 1, 1]


class _TinyPromptDataset(Dataset):
    def __init__(self):
        self.images = []
        self.masks = []
        for class_id, location in ((1, (0, 0)), (3, (0, 4)), (4, (4, 0))):
            mask = torch.zeros(8, 8, dtype=torch.long)
            row, col = location
            mask[row : row + 4, col : col + 4] = class_id
            image = torch.zeros(3, 8, 8)
            image[0] = mask.eq(1)
            image[1] = mask.eq(3)
            image[2] = mask.eq(4)
            self.images.append(image)
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]


class _TinyConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.condition_bias = nn.Parameter(torch.zeros(3))
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


def test_runner_appends_skips_and_reloads_with_injected_builders(tmp_path):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prithvi = tmp_path / "prithvi.pt"
    prithvi.write_bytes(b"tiny-prithvi")
    config["prithvi"]["checkpoint"] = str(prithvi)
    config["experiment"]["prompt_config"] = str(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml").resolve()
    )
    config["experiment"]["epochs"] = 2
    config["experiment"]["batch_size"] = 2
    config["experiment"]["seeds"] = [42]
    config["evaluation"]["preview_count"] = 4
    output = tmp_path / "rows.json"
    summary_output = tmp_path / "summary.json"
    checkpoint_dir = tmp_path / "checkpoints"
    preview_dir = tmp_path / "previews"
    build_calls = []

    def dataset_builder(_config):
        dataset = _TinyPromptDataset()
        return dataset, dataset

    def model_builder(_config, method, device):
        build_calls.append(method)
        return _TinyConditionalModel().to(device)

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
    assert len(list(checkpoint_dir.glob("*.pt"))) == 2
    assert all(row.get("synthetic_fallback") is not True for row in rows)
    prompt_rows = [row for row in rows if row["method"].startswith("siglip")]
    baseline_rows = [row for row in rows if row["method"].startswith("no_text")]
    assert all("correct_iou_by_sample" in row for row in prompt_rows)
    assert all("correct_iou_by_sample" not in row for row in baseline_rows)
    assert all(row["checkpoint_reproduced"] is True for row in rows)
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


def test_runner_resumes_an_incomplete_epoch_checkpoint(monkeypatch, tmp_path):
    import geoadapter.bench.run_geovlm_prompt_segmentation as runner

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prithvi = tmp_path / "prithvi.pt"
    prithvi.write_bytes(b"tiny-prithvi")
    config["prithvi"]["checkpoint"] = str(prithvi)
    config["experiment"]["prompt_config"] = str(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml").resolve()
    )
    config["experiment"]["epochs"] = 2
    config["experiment"]["batch_size"] = 2
    config["evaluation"]["preview_count"] = 0
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / f"{BASELINE_METHOD}__seed123.pt"

    def model_builder(_config, _method, device):
        return _TinyConditionalModel().to(device)

    trainer = build_trainer(model_builder(config, BASELINE_METHOD, "cpu"), config, "cpu")
    state = trainer.state_dict(
        epoch=1,
        metadata=checkpoint_metadata(config, BASELINE_METHOD, 123),
    )
    state["loss_history"] = [2.0]
    torch.save(state, checkpoint_path)
    train_calls = []

    def fake_train_one_epoch(
        _trainer,
        _loader,
        _prompt_config,
        _weights,
        epoch_seed,
        _method,
        *,
        empty_target_cap,
    ):
        train_calls.append((epoch_seed, empty_target_cap))
        return 1.0

    monkeypatch.setattr(runner, "_train_one_epoch", fake_train_one_epoch)
    dataset = _TinyPromptDataset()

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

    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert train_calls == [(124, 0.25)]
    assert resumed["epoch"] == 2
    assert resumed["loss_history"] == [2.0, 1.0]
    assert all(row["loss_history"] == [2.0, 1.0] for row in rows)
