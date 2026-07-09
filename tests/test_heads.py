import pytest
import torch
from geoadapter.models.heads import ClassificationHead, MultiLabelHead, SegmentationHead


class TestClassificationHead:
    def test_output_shape(self):
        head = ClassificationHead(in_dim=768, num_classes=10)
        x = torch.randn(4, 768)
        logits = head(x)
        assert logits.shape == (4, 10)


class TestMultiLabelHead:
    def test_output_shape(self):
        head = MultiLabelHead(in_dim=768, num_classes=19)
        x = torch.randn(4, 768)
        logits = head(x)
        assert logits.shape == (4, 19)


class TestSegmentationHead:
    def test_output_shape(self):
        head = SegmentationHead(in_dim=768, num_classes=7, patch_size=16)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 7, 16, 16)

    def test_patch_token_output_shape(self):
        head = SegmentationHead(in_dim=768, num_classes=7, patch_size=16)
        x = torch.randn(4, 4, 768)
        out = head(x, spatial_dims=(2, 2))
        assert out.shape == (4, 7, 32, 32)

    def test_default_linear_decoder_param_count_is_unchanged(self):
        head = SegmentationHead(in_dim=768, num_classes=6, patch_size=16)
        n_params = sum(p.numel() for p in head.parameters())

        assert n_params == 768 * 6 + 6

    def test_conv_lite_decoder_matches_linear_output_shape_with_more_capacity(self):
        linear = SegmentationHead(in_dim=32, num_classes=5, patch_size=8)
        conv_lite = SegmentationHead(
            in_dim=32,
            num_classes=5,
            patch_size=8,
            decoder_type="conv_lite",
            hidden_dim=16,
        )
        x = torch.randn(2, 9, 32)

        out = conv_lite(x, spatial_dims=(3, 3))

        assert out.shape == (2, 5, 24, 24)
        assert sum(p.numel() for p in conv_lite.parameters()) > sum(
            p.numel() for p in linear.parameters()
        )

    def test_rejects_unknown_decoder_type(self):
        with pytest.raises(ValueError, match="unknown segmentation decoder"):
            SegmentationHead(decoder_type="not_a_decoder")
