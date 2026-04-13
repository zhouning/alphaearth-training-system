# %% [markdown]
# # Private GF-2 Case Study: Embedding Visualization
# Qualitative analysis — no labels needed.
#
# Install: `pip install -e /content/AlphaEarth-System/[bench]`

# %% Imports
import torch
import numpy as np
from geoadapter.models.prithvi import PrithviBackbone
from geoadapter.adapters.geo_adapter import GeoAdapter
from geoadapter.adapters.zero_pad import ZeroPadAdapter
from geoadapter.viz import compute_tsne, extract_channel_attention_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% Extract embeddings with GeoAdapter vs ZeroPad
# Load your trained GeoAdapter checkpoint here
# Compare t-SNE plots of embeddings from both approaches
print("Load private GF-2 data and trained checkpoints to generate t-SNE plots.")
