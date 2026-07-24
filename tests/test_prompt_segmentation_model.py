import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from geoadapter.models.prompt_segmentation import (
    PromptSegmentationModel,
    ThreeHeadSegmentationBaseline,
)


class FakeTextEncoder(nn.Module):
    output_dim = 6

    def forward(self, prompts):
        rows = []
        for prompt in prompts:
            code = sum(ord(char) for char in prompt)
            rows.append(
                torch.tensor([(code + index * 17) % 31 for index in range(6)]).float()
            )
        return torch.stack(rows)


class FrozenFakeTextEncoder(FakeTextEncoder):
    def __init__(self):
        super().__init__()
        self.frozen_offset = nn.Parameter(torch.zeros(6), requires_grad=False)

    def forward(self, prompts):
        return super().forward(prompts) + self.frozen_offset


class TinySpatialBackbone(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 8, 4, stride=4)

    def forward(self, images, return_spatial=False):
        features = self.conv(images)
        h, w = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return (tokens, (h, w)) if return_spatial else tokens.mean(dim=1)


def test_prompt_model_output_shape_and_prompt_dependence():
    torch.manual_seed(2)
    model = PromptSegmentationModel(
        TinySpatialBackbone(),
        FakeTextEncoder(),
        visual_dim=8,
        text_dim=6,
        condition_dim=5,
        decoder_dim=7,
        patch_size=4,
    )
    images = torch.randn(2, 6, 16, 20)
    first = model(images, ["find buildings", "find buildings"])
    second = model(images, ["find water", "find water"])
    assert first.shape == (2, 16, 20)
    assert not torch.allclose(first, second)


def test_three_head_baseline_selects_requested_target_channel():
    model = ThreeHeadSegmentationBaseline(
        TinySpatialBackbone(), visual_dim=8, decoder_dim=7, patch_size=4
    )
    logits = model(torch.randn(2, 6, 16, 20), torch.tensor([1, 4]))
    assert logits.shape == (2, 16, 20)


def test_prompt_model_gradients_reach_conditioning_but_not_frozen_text():
    model = PromptSegmentationModel(
        TinySpatialBackbone(),
        FrozenFakeTextEncoder(),
        visual_dim=8,
        text_dim=6,
        condition_dim=5,
        decoder_dim=7,
        patch_size=4,
    )
    images = torch.randn(2, 6, 16, 20)

    model(images, ["find buildings", "find water"]).mean().backward()

    for module in (
        model.visual_similarity,
        model.text_projection,
        model.film,
        model.decoder,
    ):
        assert all(parameter.grad is not None for parameter in module.parameters())
    assert model.logit_scale.grad is not None
    assert model.text_encoder.frozen_offset.grad is None


def test_siglip_text_encoder_freezes_tower_and_keeps_it_in_eval(monkeypatch):
    loader_calls = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            loader_calls.append(("tokenizer", args, kwargs))
            return cls()

        def __call__(self, prompts, **kwargs):
            return {"input_ids": torch.ones(len(prompts), 3, dtype=torch.long)}

    class FakeSiglipTextModel(nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            loader_calls.append(("model", args, kwargs))
            return cls()

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.config = types.SimpleNamespace(hidden_size=4)

        def forward(self, input_ids):
            pooled = self.weight.unsqueeze(0).expand(input_ids.shape[0], -1)
            return types.SimpleNamespace(pooler_output=pooled)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=FakeTokenizer,
            SiglipTextModel=FakeSiglipTextModel,
        ),
    )
    text_encoder_module = importlib.import_module("geoadapter.models.text_encoder")
    model_id = "local/test-siglip"
    revision = "seed42-recovery"
    cache_dir = Path("models/huggingface-cache")
    encoder = text_encoder_module.SiglipTextEncoder(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )

    expected_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": True,
    }
    assert loader_calls == [
        ("tokenizer", (model_id,), expected_kwargs),
        ("model", (model_id,), expected_kwargs),
    ]
    assert encoder.output_dim == 4
    assert encoder(["find buildings", "find water"]).shape == (2, 4)
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    encoder.train()
    assert encoder.model.training is False
    with pytest.raises(ValueError, match="non-empty"):
        encoder([""])
