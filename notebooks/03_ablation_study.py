# %% [markdown]
# # GeoAdapter Ablation Study
# Compare: Full GeoAdapter vs Proj+SE vs Proj-only vs Zero-pad
#
# Install: `pip install -e /content/AlphaEarth-System/[bench]`

# %% Imports
import torch
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.adapters.zero_pad import ZeroPadAdapter
from geoadapter.models.heads import ClassificationHead
from geoadapter.engine.trainer import PEFTTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% Define ablation variants
# Full GeoAdapter: Proj + SE + DWConv (default)
# Proj+SE only: remove spatial_refine
# Proj only: remove channel_attn and spatial_refine
# Zero-pad: no adapter
print("Ablation variants defined. Load dataset and run training loop below.")
