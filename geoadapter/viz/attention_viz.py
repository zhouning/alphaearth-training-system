"""Channel attention heatmap visualization for GeoAdapter."""
import numpy as np


def plot_channel_attention_heatmap(adapter, modality_labels=None, save_path=None):
    """Generate and optionally save a channel attention weight heatmap.

    Args:
        adapter: A GeoAdapter instance.
        modality_labels: List of input band names (e.g., ["B", "G", "R", "NIR"]).
        save_path: If provided, save the figure to this path.

    Returns:
        attention_weights: numpy array of shape [out_channels].
    """
    import torch

    adapter.eval()
    dummy = torch.randn(1, adapter.in_channels, 64, 64)
    with torch.no_grad():
        proj = adapter.channel_proj(dummy)
        attn = adapter.channel_attn(proj)
    weights = attn.squeeze().cpu().numpy()

    if save_path:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.imshow(weights.reshape(1, -1), cmap="YlOrRd", aspect="auto")
            ax.set_yticks([])
            out_labels = [f"Ch{i}" for i in range(len(weights))]
            ax.set_xticks(range(len(weights)))
            ax.set_xticklabels(out_labels)
            ax.set_title("GeoAdapter Channel Attention Weights")
            if modality_labels:
                ax.set_xlabel(f"Input: {', '.join(modality_labels)}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except ImportError:
            pass

    return weights
