# %% [markdown]
# # Geo-MLOps System Evaluation
# Measure end-to-end latency and throughput.

# %% Imports
import time
import torch
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.models.heads import ClassificationHead
from geoadapter.engine.trainer import PEFTTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Throughput benchmark
backbone = PrithviBackbone(pretrained=False)
adapter = GeoAdapter(in_channels=4, out_channels=6)
head = ClassificationHead(in_dim=768, num_classes=10)
trainer = PEFTTrainer(backbone, adapter, head, lr=1e-3, device=device)

# Warm up
x = torch.randn(16, 4, 64, 64)
y = torch.randint(0, 10, (16,))
trainer.train_step(x, y)

# Measure
start = time.time()
for _ in range(10):
    trainer.train_step(x, y)
elapsed = time.time() - start
print(f"10 steps in {elapsed:.2f}s = {10/elapsed:.1f} steps/sec")
print(f"Throughput: {16 * 10 / elapsed:.0f} samples/sec")
