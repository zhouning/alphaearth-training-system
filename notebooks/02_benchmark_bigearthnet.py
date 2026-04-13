# %% [markdown]
# # GeoAdapter Benchmark: BigEarthNet-S2 (Multi-Label)
# Run on Google Colab Pro+ with A100 GPU.
#
# Install: `pip install -e /content/AlphaEarth-System/[bench]`

# %% Imports
import torch
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters import GeoAdapter, ZeroPadAdapter
from geoadapter.models.heads import MultiLabelHead
from geoadapter.engine.trainer import PEFTTrainer
from geoadapter.engine.evaluator import compute_multilabel_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% Smoke test with multilabel task
backbone = PrithviBackbone(pretrained=False)
adapter = GeoAdapter(in_channels=4, out_channels=6)
head = MultiLabelHead(in_dim=768, num_classes=19)
trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3, task="multilabel", device=device)

x = torch.randn(4, 4, 64, 64)
y = torch.zeros(4, 19)
y[:, :3] = 1.0  # dummy multilabel targets
loss = trainer.train_step(x, y)
print(f"Smoke test loss: {loss:.4f}")
