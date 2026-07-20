import pytest
import torch

from geoadapter.models.prithvi import PrithviBackbone


def _position_tensor(embed_dim=8):
    return torch.arange(589 * embed_dim, dtype=torch.float32).reshape(
        1, 589, embed_dim
    )


def test_checkpoint_positions_reduce_three_temporal_grids_and_interpolate():
    model = PrithviBackbone(
        pretrained=False,
        embed_dim=8,
        depth=1,
        num_heads=2,
        use_checkpoint_position_embeddings=True,
    )
    position = _position_tensor()
    model.set_checkpoint_position_embedding(position)
    cls, patch = model.interpolate_checkpoint_positions((8, 10))
    assert cls.shape == (1, 1, 8)
    assert patch.shape == (1, 80, 8)
    assert torch.equal(cls, position[:, :1])
    expected_grid = (
        position[:, 1:]
        .reshape(1, 3, 14, 14, 8)
        .mean(dim=1)
        .permute(0, 3, 1, 2)
    )
    assert torch.equal(model.checkpoint_patch_positions, expected_grid)


def test_checkpoint_positions_reject_unexpected_token_count():
    model = PrithviBackbone(
        pretrained=False,
        embed_dim=8,
        depth=1,
        num_heads=2,
        use_checkpoint_position_embeddings=True,
    )
    with pytest.raises(ValueError, match="589"):
        model.set_checkpoint_position_embedding(torch.zeros(1, 590, 8))


def test_default_path_remains_position_free():
    model = PrithviBackbone(
        pretrained=False,
        embed_dim=8,
        depth=1,
        num_heads=2,
    )
    assert model.use_checkpoint_position_embeddings is False


def test_checkpoint_load_populates_positions_only_when_enabled(tmp_path):
    path = tmp_path / "tiny_prithvi.pt"
    torch.save({"model": {"encoder.pos_embed": _position_tensor()}}, path)

    enabled = PrithviBackbone(
        checkpoint_path=str(path),
        embed_dim=8,
        depth=1,
        num_heads=2,
        use_checkpoint_position_embeddings=True,
    )
    disabled = PrithviBackbone(
        checkpoint_path=str(path),
        embed_dim=8,
        depth=1,
        num_heads=2,
    )

    assert enabled.checkpoint_cls_position.shape == (1, 1, 8)
    assert enabled.checkpoint_patch_positions.shape == (1, 8, 14, 14)
    assert disabled.checkpoint_cls_position.numel() == 0
    assert disabled.checkpoint_patch_positions.numel() == 0


def test_opt_in_checkpoint_load_rejects_incompatible_positions(tmp_path):
    path = tmp_path / "bad_prithvi.pt"
    torch.save({"model": {"encoder.pos_embed": torch.zeros(1, 590, 8)}}, path)

    with pytest.raises(ValueError, match="589"):
        PrithviBackbone(
            checkpoint_path=str(path),
            embed_dim=8,
            depth=1,
            num_heads=2,
            use_checkpoint_position_embeddings=True,
        )


def test_opt_in_checkpoint_load_requires_positions(tmp_path):
    path = tmp_path / "missing_positions.pt"
    torch.save({"model": {}}, path)

    with pytest.raises(ValueError, match="encoder.pos_embed"):
        PrithviBackbone(
            checkpoint_path=str(path),
            embed_dim=8,
            depth=1,
            num_heads=2,
            use_checkpoint_position_embeddings=True,
        )
