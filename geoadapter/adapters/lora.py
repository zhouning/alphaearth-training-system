import torch
import torch.nn as nn
import math


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
