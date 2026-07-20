import pytest
import torch
import torch.nn as nn

from geoadapter.engine.prompt_segmentation import (
    PromptSegmentationLoss,
    PromptSegmentationTrainer,
)
from geoadapter.models.prompt_segmentation import PromptSegmentationModel


class _FakeTextEncoder(nn.Module):
    output_dim = 4

    def forward(self, prompts):
        rows = []
        for prompt in prompts:
            code = sum(ord(char) for char in prompt)
            rows.append(torch.tensor([(code + index * 7) % 19 for index in range(4)]))
        return torch.stack(rows).float()


class _TinySpatialBackbone(nn.Module):
    embed_dim = 6

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 6, 4, stride=4)

    def forward(self, images, return_spatial=False):
        features = self.conv(images)
        spatial_dims = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return (tokens, spatial_dims) if return_spatial else tokens.mean(dim=1)


def _build_model():
    return PromptSegmentationModel(
        _TinySpatialBackbone(),
        _FakeTextEncoder(),
        visual_dim=6,
        text_dim=4,
        condition_dim=4,
        decoder_dim=6,
        patch_size=4,
    )


def test_prompt_loss_is_lower_for_correct_logits():
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    correct = torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]])
    wrong = -correct
    loss = PromptSegmentationLoss()
    assert loss(correct, targets, torch.tensor([2.0])) < loss(
        wrong, targets, torch.tensor([2.0])
    )


def test_prompt_loss_is_finite_for_empty_target():
    value = PromptSegmentationLoss()(
        torch.zeros(2, 4, 4), torch.zeros(2, 4, 4), torch.ones(2)
    )
    assert torch.isfinite(value)


def test_prompt_loss_requires_one_positive_weight_per_example():
    with pytest.raises(ValueError, match="positive_weights"):
        PromptSegmentationLoss()(
            torch.zeros(2, 4, 4),
            torch.zeros(2, 4, 4),
            torch.ones(1),
        )


def test_prompt_trainer_reloads_identical_predictions_after_loss_decreases():
    torch.manual_seed(11)
    images = torch.zeros(4, 6, 8, 8)
    images[0, :, :4, :4] = 1
    images[1, :, :4, 4:] = 1
    images[2, :, 4:, :4] = 1
    images[3, :, 4:, 4:] = 1
    targets = images[:, 0]
    prompts = ["find buildings", "find water", "find roads", "find buildings"]
    positive_weights = torch.ones(4)
    trainer = PromptSegmentationTrainer(
        _build_model(), lr=1e-2, epochs=20, device="cpu"
    )

    first = trainer.train_step(images, prompts, targets, positive_weights)
    for _ in range(19):
        last = trainer.train_step(images, prompts, targets, positive_weights)
    assert last < first

    conditions = ["find buildings"] * len(images)
    before = trainer.predict(images, conditions)
    state = trainer.state_dict(epoch=20, metadata={"schema": "test"})
    torch.manual_seed(29)
    clone = PromptSegmentationTrainer(
        _build_model(), lr=1e-2, epochs=20, device="cpu"
    )
    epoch, metadata = clone.load_state_dict(state)
    after = clone.predict(images, conditions)

    assert epoch == 20
    assert metadata == {"schema": "test"}
    assert torch.allclose(before, after, atol=1e-6)


def test_prompt_trainer_rejects_missing_trainable_checkpoint_tensor():
    trainer = PromptSegmentationTrainer(
        _build_model(), lr=1e-2, epochs=2, device="cpu"
    )
    state = trainer.state_dict(epoch=1, metadata={"schema": "test"})
    removed_name = next(iter(state["trainable_model"]))
    del state["trainable_model"][removed_name]
    clone = PromptSegmentationTrainer(
        _build_model(), lr=1e-2, epochs=2, device="cpu"
    )

    with pytest.raises(ValueError, match="missing trainable checkpoint parameters"):
        clone.load_state_dict(state)
