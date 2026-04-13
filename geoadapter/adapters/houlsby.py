import torch
import torch.nn as nn


class HoulsbyBottleneck(nn.Module):
    """Bottleneck adapter inserted after FFN in a Transformer layer."""

    def __init__(self, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


def inject_houlsby_adapters(block: nn.Module, bottleneck_dim: int = 64):
    """Wrap the forward of a TransformerEncoderLayer to insert adapter after FFN."""
    # Detect d_model from the block's layer norm
    d_model = 768
    if hasattr(block, "norm1"):
        d_model = block.norm1.normalized_shape[0]
    elif hasattr(block, "self_attn") and hasattr(block.self_attn, "embed_dim"):
        d_model = block.self_attn.embed_dim

    adapter = HoulsbyBottleneck(d_model=d_model, bottleneck_dim=bottleneck_dim)
    original_forward = block.forward

    def new_forward(src, *args, **kwargs):
        out = original_forward(src, *args, **kwargs)
        return adapter(out)

    block.forward = new_forward
    block.add_module("houlsby_adapter", adapter)
    for name, p in block.named_parameters():
        p.requires_grad_("houlsby_adapter" in name)
