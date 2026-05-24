import torch.nn as nn


def configure_bitfit(module: nn.Module):
    """Freeze all parameters except biases.

    LayerNorm weights are also unfrozen as a PyTorch autograd correctness
    workaround: if a LayerNorm has bias.requires_grad=True but
    weight.requires_grad=False, native_layer_norm_backward returns an empty
    tensor for the weight slot, which the autograd engine flags as a shape
    mismatch (`expected [embed_dim], got [0]`). Including the LN affine
    weight is also standard practice for transformer-style BitFit (see the
    original Ben Zaken et al. 2022 reference implementation for ViT/BERT).
    The added parameters are negligible (~24 LN layers x 768 dim ~= 18K).
    """
    for name, param in module.named_parameters():
        param.requires_grad_("bias" in name)
    for m in module.modules():
        if isinstance(m, nn.LayerNorm) and m.weight is not None:
            m.weight.requires_grad_(True)
