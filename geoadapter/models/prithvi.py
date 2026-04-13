import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PrithviBackbone(nn.Module):
    """Full Prithvi-100M ViT backbone with frozen weights.

    Architecture: Conv2d patch embed (from Conv3d squeeze) + 12 Transformer layers.
    Prithvi-100M config: embed_dim=768, depth=12, num_heads=12, patch_size=16, in_chans=6.
    """

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        in_chans: int = 6,
        patch_size: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding: Conv3d(6,768,(1,16,16)) squeezed to Conv2d
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, batch_first=True,
                activation="gelu", norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if pretrained and checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self._freeze_all()

    def _load_checkpoint(self, path: str):
        """Load Prithvi-100M weights, squeezing Conv3d temporal dim."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state = ckpt.get("model", ckpt)

            # Patch embed: Conv3d [768,6,1,16,16] -> Conv2d [768,6,16,16]
            pe_key = "encoder.patch_embed.proj.weight"
            if pe_key in state and state[pe_key].dim() == 5:
                state[pe_key] = state[pe_key].squeeze(2)

            # Map keys and load what matches
            own_state = self.state_dict()
            loaded = 0
            for k, v in state.items():
                mapped = k.replace("encoder.", "", 1) if k.startswith("encoder.") else k
                if mapped in own_state and own_state[mapped].shape == v.shape:
                    own_state[mapped] = v
                    loaded += 1
            self.load_state_dict(own_state, strict=False)
            logger.info(f"Loaded {loaded} tensors from Prithvi checkpoint")
        except Exception as e:
            logger.warning(f"Could not load Prithvi weights: {e}")

    def _freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 6, H, W] -> [B, 768] global features."""
        B = x.shape[0]
        x = self.patch_embed(x)                    # [B, 768, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)           # [B, N, 768]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)             # [B, N+1, 768]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]                             # CLS token -> [B, 768]
