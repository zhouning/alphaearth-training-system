import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import torch.nn as nn
import yaml
from PIL import Image
from rasterio.transform import from_origin

from geoadapter.bench.run_geovlm_prompt_segmentation import (
    BASELINE_METHOD,
    PROMPT_METHOD,
    checkpoint_metadata,
    probe_rank,
)
from geoadapter.inference.prompt_segmentation import (
    load_prompt_checkpoint,
    load_prompt_image,
    predict_prompt_image,
    run_prompt_inference,
)


CONFIG_PATH = Path("geoadapter/bench/configs/geovlm_prompt_segmentation.yaml")


class _TinyPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(-4.0))

    def forward(self, images, prompts):
        assert len(prompts) == images.shape[0]
        return images[:, 0] * self.scale + self.bias


def _finite_probe(change=0.5, loss=1.0):
    return {
        "finite": True,
        "mean_loss": float(loss),
        "nonconstant_class_count": 3,
        "prompt_map_changed_class_count": 3,
        "mean_prompt_probability_change": float(change),
        "classes": {
            class_name: {
                "prediction_range": 1.0,
                "prediction_nonconstant": True,
                "mean_prompt_probability_change": float(change),
                "prompt_map_changed": True,
            }
            for class_name in ("building", "water", "road")
        },
    }


def _checkpoint_fixture(tmp_path, *, method=PROMPT_METHOD):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    prithvi = tmp_path / "Prithvi_100M.pt"
    prithvi.write_bytes(b"tiny-prithvi")
    config["prithvi"]["checkpoint"] = str(prithvi)
    config["experiment"]["prompt_config"] = str(
        Path("geoadapter/bench/configs/geovlm_prompts.yaml").resolve()
    )
    config["text_encoder"]["revision"] = "fixture-siglip-sha"
    model = _TinyPromptModel()
    checkpoint = tmp_path / "prompt.best.pt"
    probe = {
        **_finite_probe(),
        "epoch": 1,
        "training_loss": 1.0,
    }
    best_rank = list(probe_rank(probe))
    metadata = checkpoint_metadata(config, method, 42)
    metadata.update({"best_epoch": 1, "best_probe_rank": best_rank})
    torch.save(
        {
            "metadata": metadata,
            "checkpoint_role": "best",
            "epoch": 1,
            "best_epoch": 1,
            "best_probe_rank": best_rank,
            "loss_history": [1.0],
            "probe_history": [probe],
            "positive_weights": {"building": 2.0, "water": 3.0, "road": 4.0},
            "trainable_model": {
                name: value.detach().clone()
                for name, value in model.named_parameters()
            },
        },
        checkpoint,
    )
    return config, checkpoint


def _model_builder(_config, _method, device):
    return _TinyPromptModel().to(device)


def test_png_inference_writes_mask_probability_preview_and_metadata(tmp_path):
    config, checkpoint = _checkpoint_fixture(tmp_path)
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[:, :5, 0] = 255
    image_path = tmp_path / "sample.png"
    Image.fromarray(rgb).save(image_path)

    metadata = predict_prompt_image(
        image_path,
        "segment all buildings",
        checkpoint,
        tmp_path / "outputs",
        config=config,
        threshold=0.5,
        device="cpu",
        model_builder=_model_builder,
    )

    output_dir = tmp_path / "outputs"
    mask_path = output_dir / "sample_mask.png"
    probability_path = output_dir / "sample_probability.npy"
    preview_path = output_dir / "sample_preview.png"
    metadata_path = output_dir / "sample_metadata.json"
    assert mask_path.is_file()
    assert probability_path.is_file()
    assert preview_path.is_file()
    assert metadata_path.is_file()
    assert np.load(probability_path).shape == (8, 10)
    assert set(np.unique(np.asarray(Image.open(mask_path)))) <= {0, 255}
    assert metadata == json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["prompt"] == "segment all buildings"
    assert metadata["validated_semantic_scope"] == ["building", "road", "water"]
    assert metadata["threshold"] == 0.5
    assert metadata["checkpoint_schema"] == "paper12.geovlm_prompt_checkpoint.v2"
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_state["metadata"]["training_contract"] == (
        "paper12.geovlm_prompt_training.v2"
    )
    assert metadata["prithvi_sha256"]
    assert metadata["prompt_config_sha256"]
    assert metadata["georeferenced"] is False
    assert metadata["output_paths"]["mask"] == str(mask_path)


def test_geotiff_inference_preserves_crs_transform_and_shape(tmp_path):
    config, checkpoint = _checkpoint_fixture(tmp_path)
    image_path = tmp_path / "projected.tif"
    transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)
    rgb = np.zeros((3, 6, 7), dtype=np.uint8)
    rgb[0, :, :3] = 255
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=7,
        height=6,
        count=3,
        dtype="uint8",
        crs="EPSG:32650",
        transform=transform,
    ) as dst:
        dst.write(rgb)

    metadata = predict_prompt_image(
        image_path,
        "identify roofed structures",
        checkpoint,
        tmp_path / "outputs",
        config=config,
        device="cpu",
        model_builder=_model_builder,
    )

    mask_path = tmp_path / "outputs" / "projected_mask.tif"
    with rasterio.open(mask_path) as mask:
        assert mask.count == 1
        assert mask.dtypes == ("uint8",)
        assert mask.crs == rasterio.crs.CRS.from_epsg(32650)
        assert mask.transform == transform
        assert (mask.height, mask.width) == (6, 7)
        assert set(np.unique(mask.read(1))) <= {0, 1}
    assert metadata["georeferenced"] is True
    assert metadata["crs"] == "EPSG:32650"


def test_prompt_inference_validates_prompt_threshold_and_extension(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
    image = load_prompt_image(image_path)
    model = _TinyPromptModel()

    with pytest.raises(ValueError, match="non-empty"):
        run_prompt_inference(model, image, " ")
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_prompt_inference(model, image, "water", threshold=1.1)
    with pytest.raises(ValueError, match="unsupported image extension"):
        load_prompt_image(tmp_path / "sample.bmp")


def test_checkpoint_loader_rejects_baseline_and_hash_mismatch(tmp_path):
    config, checkpoint = _checkpoint_fixture(tmp_path, method=BASELINE_METHOD)
    with pytest.raises(ValueError, match="prompt checkpoint"):
        load_prompt_checkpoint(
            checkpoint,
            config,
            device="cpu",
            model_builder=_model_builder,
        )

    config, checkpoint = _checkpoint_fixture(tmp_path)
    Path(config["prithvi"]["checkpoint"]).write_bytes(b"different")
    with pytest.raises(ValueError, match="prithvi_sha256"):
        load_prompt_checkpoint(
            checkpoint,
            config,
            device="cpu",
            model_builder=_model_builder,
        )


def test_checkpoint_loader_restores_missing_config_revision_from_metadata(
    tmp_path,
):
    config, checkpoint = _checkpoint_fixture(tmp_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["metadata"]["siglip_revision"] = "checkpoint-resolved-sha"
    torch.save(state, checkpoint)
    config["text_encoder"]["revision"] = None
    captured = {}

    def capturing_builder(resolved_config, _method, device):
        captured["revision"] = resolved_config["text_encoder"]["revision"]
        return _TinyPromptModel().to(device)

    load_prompt_checkpoint(
        checkpoint,
        config,
        device="cpu",
        model_builder=capturing_builder,
    )

    assert captured["revision"] == "checkpoint-resolved-sha"


def test_checkpoint_loader_rejects_explicit_revision_drift(tmp_path):
    config, checkpoint = _checkpoint_fixture(tmp_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["metadata"]["siglip_revision"] = "checkpoint-resolved-sha"
    torch.save(state, checkpoint)
    config["text_encoder"]["revision"] = "different-explicit-sha"

    with pytest.raises(ValueError, match="siglip_revision"):
        load_prompt_checkpoint(
            checkpoint,
            config,
            device="cpu",
            model_builder=_model_builder,
        )


def test_checkpoint_loader_rejects_last_checkpoint_even_with_best_metadata(
    tmp_path,
):
    config, checkpoint = _checkpoint_fixture(tmp_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["checkpoint_role"] = "last"
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="best checkpoint.*role"):
        load_prompt_checkpoint(
            checkpoint,
            config,
            device="cpu",
            model_builder=_model_builder,
        )


@pytest.mark.parametrize("forgery", ["metadata", "history"])
def test_checkpoint_loader_rejects_forged_best_selection_contract(
    tmp_path, forgery
):
    config, checkpoint = _checkpoint_fixture(tmp_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if forgery == "metadata":
        state["metadata"]["best_epoch"] = 2
    else:
        second_probe = {
            **_finite_probe(change=0.1, loss=2.0),
            "epoch": 2,
            "training_loss": 2.0,
        }
        second_rank = list(probe_rank(second_probe))
        state.update(
            {
                "epoch": 2,
                "best_epoch": 2,
                "best_probe_rank": second_rank,
                "loss_history": [1.0, 2.0],
                "probe_history": [state["probe_history"][0], second_probe],
            }
        )
        state["metadata"].update(
            {"best_epoch": 2, "best_probe_rank": second_rank}
        )
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="best selection"):
        load_prompt_checkpoint(
            checkpoint,
            config,
            device="cpu",
            model_builder=_model_builder,
        )
