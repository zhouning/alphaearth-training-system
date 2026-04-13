# %% [markdown]
# # GeoAdapter Benchmark: EuroSAT
# Run on Google Colab Pro+ with A100 GPU.
#
# Install: `pip install -e /content/AlphaEarth-System/[bench]`

# %% Imports
import torch
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters import GeoAdapter, ZeroPadAdapter
from geoadapter.models.heads import ClassificationHead
from geoadapter.engine.trainer import PEFTTrainer
from geoadapter.engine.evaluator import compute_classification_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% Quick smoke test
backbone = PrithviBackbone(pretrained=False)
adapter = GeoAdapter(in_channels=3, out_channels=6)
head = ClassificationHead(in_dim=768, num_classes=10)
trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3, device=device)

x = torch.randn(4, 3, 64, 64)
y = torch.randint(0, 10, (4,))
loss = trainer.train_step(x, y)
print(f"Smoke test loss: {loss:.4f}")
print("GeoAdapter package loaded successfully!")
