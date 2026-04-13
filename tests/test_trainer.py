import torch
from geoadapter.engine.trainer import PEFTTrainer
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.models.heads import ClassificationHead


class TestPEFTTrainer:
    def test_one_step(self):
        backbone = PrithviBackbone(pretrained=False)
        adapter = GeoAdapter(in_channels=4, out_channels=6)
        head = ClassificationHead(in_dim=768, num_classes=10)
        trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3)

        x = torch.randn(4, 4, 128, 128)
        y = torch.randint(0, 10, (4,))
        loss = trainer.train_step(x, y)
        assert isinstance(loss, float)
        assert loss > 0

    def test_predict_shape(self):
        backbone = PrithviBackbone(pretrained=False)
        adapter = GeoAdapter(in_channels=3, out_channels=6)
        head = ClassificationHead(in_dim=768, num_classes=10)
        trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3)

        x = torch.randn(2, 3, 64, 64)
        logits = trainer.predict(x)
        assert logits.shape == (2, 10)
