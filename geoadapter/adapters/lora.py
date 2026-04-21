import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 8):
        super().__init__()
        self.original = original
        self.rank = rank
        d_in, d_out = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    def forward(self, x):
        base = self.original(x)
        return base + (x @ self.lora_A @ self.lora_B)


def inject_lora(module: nn.Module, rank: int = 8, target_modules=("self_attn",)):
    """Replace Linear layers inside target submodules with LoRALinear."""
    for tgt in target_modules:
        if hasattr(module, tgt):
            submod = getattr(module, tgt)
            for name, child in list(submod.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(submod, name, LoRALinear(child, rank=rank))
    # Freeze everything except LoRA params
    for p in module.parameters():
        p.requires_grad_(False)
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)


def remove_lora(module: nn.Module):
    """Merge LoRA weights back into original Linear and remove wrappers."""
    for name, child in list(module.named_modules()):
        if isinstance(child, LoRALinear):
            merged = child.original
            merged.weight.data += (child.lora_A @ child.lora_B).T
            merged.weight.requires_grad_(True)
            parts = name.split(".")
            parent = module
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], merged)


def _make_split_qkv_forward(mha, q_proj, k_proj, v_proj):
    """Build a manual MHA forward that uses separate Q/K/V Linear modules."""
    num_heads = mha.num_heads
    head_dim = mha.head_dim
    out_proj = mha.out_proj

    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B, T, D = query.shape
        S = key.shape[1]
        q = q_proj(query).view(B, T, num_heads, head_dim).transpose(1, 2)
        k = k_proj(key).view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v_proj(value).view(B, S, num_heads, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn_weights = attn_weights.softmax(dim=-1)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)
        return out_proj(out), None

    return forward


def split_qkv_and_inject_lora(block: nn.Module, rank: int = 8):
    """Split fused QKV into separate Q/K/V nn.Linear, then inject LoRA on all projections.

    PyTorch's MultiheadAttention stores Q/K/V as a single in_proj_weight parameter,
    not as child nn.Linear modules. inject_lora() can't find them. This function
    splits them into real modules so LoRA can wrap each one.
    """
    mha = block.self_attn
    embed_dim = mha.embed_dim

    w = mha.in_proj_weight.data
    b = mha.in_proj_bias.data if mha.in_proj_bias is not None else None

    q_proj = nn.Linear(embed_dim, embed_dim)
    k_proj = nn.Linear(embed_dim, embed_dim)
    v_proj = nn.Linear(embed_dim, embed_dim)

    q_proj.weight.data.copy_(w[:embed_dim])
    k_proj.weight.data.copy_(w[embed_dim : 2 * embed_dim])
    v_proj.weight.data.copy_(w[2 * embed_dim :])

    if b is not None:
        q_proj.bias.data.copy_(b[:embed_dim])
        k_proj.bias.data.copy_(b[embed_dim : 2 * embed_dim])
        v_proj.bias.data.copy_(b[2 * embed_dim :])

    mha.q_proj = q_proj
    mha.k_proj = k_proj
    mha.v_proj = v_proj

    mha.forward = _make_split_qkv_forward(mha, q_proj, k_proj, v_proj)

    inject_lora(block, rank=rank, target_modules=("self_attn",))
