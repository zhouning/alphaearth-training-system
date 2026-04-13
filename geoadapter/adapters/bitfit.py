import torch.nn as nn


def configure_bitfit(module: nn.Module):
    """Freeze all parameters except biases."""
    for name, param in module.named_parameters():
        param.requires_grad_("bias" in name)
