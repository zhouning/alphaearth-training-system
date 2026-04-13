import torch
from geoadapter.models.heads import ClassificationHead, MultiLabelHead


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
