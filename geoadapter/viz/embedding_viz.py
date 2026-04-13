"""Embedding visualization: t-SNE/UMAP + channel attention heatmaps."""
import numpy as np


def compute_tsne(embeddings: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """Reduce N x D embeddings to N x 2 via t-SNE."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)


def compute_umap(embeddings: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Reduce N x D embeddings to N x 2 via UMAP."""
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, random_state=42).fit_transform(embeddings)


def extract_channel_attention_weights(adapter) -> np.ndarray:
    """Extract SE attention weights from a GeoAdapter for visualization."""
    import torch
    adapter.eval()
    dummy = torch.randn(1, adapter.in_channels, 64, 64)
    with torch.no_grad():
        proj = adapter.channel_proj(dummy)
        attn = adapter.channel_attn(proj)
    return attn.squeeze().cpu().numpy()
