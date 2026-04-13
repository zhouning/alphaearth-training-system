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
