import pytest
import torch
from geoadapter.models.prithvi import PrithviBackbone


class TestPrithviBackbone:
    def test_load_without_weights(self):
        model = PrithviBackbone(pretrained=False)
        assert model is not None

    def test_forward_shape(self):
        model = PrithviBackbone(pretrained=False)
        x = torch.randn(2, 6, 224, 224)
        features = model(x)
        assert features.shape == (2, 768)

    def test_all_frozen(self):
        model = PrithviBackbone(pretrained=False)
        for p in model.parameters():
            assert not p.requires_grad

    def test_num_blocks(self):
        model = PrithviBackbone(pretrained=False)
        assert len(model.blocks) == 12

    @pytest.mark.parametrize("h,w", [(128, 128), (224, 224), (64, 64)])
    def test_variable_input_size(self, h, w):
        model = PrithviBackbone(pretrained=False)
        x = torch.randn(1, 6, h, w)
        features = model(x)
        assert features.shape == (1, 768)
