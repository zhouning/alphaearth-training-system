from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_CLASS_TO_CHANNEL = {1: 0, 3: 1, 4: 2}


class _BinaryDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, features, output_size):
        logits = self.net(features)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )[:, 0]


class PromptSegmentationModel(nn.Module):
    def __init__(
        self,
        backbone,
        text_encoder,
        *,
        visual_dim=768,
        text_dim=768,
        condition_dim=256,
        decoder_dim=128,
        patch_size=16,
    ):
        super().__init__()
        self.backbone = backbone
        self.text_encoder = text_encoder
        self.patch_size = patch_size
        self.visual_similarity = nn.Linear(visual_dim, condition_dim)
        self.text_projection = nn.Linear(text_dim, condition_dim)
        self.visual_decoder = nn.Linear(visual_dim, decoder_dim)
        self.film = nn.Sequential(
            nn.Linear(condition_dim, decoder_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_dim * 2, decoder_dim * 2),
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.decoder = _BinaryDecoder(decoder_dim + 1, decoder_dim)

    def forward(self, images, prompts):
        if len(prompts) != images.shape[0]:
            raise ValueError("one prompt is required per image")
        tokens, (h, w) = self.backbone(images, return_spatial=True)
        text = self.text_encoder(prompts).to(tokens.device)
        visual_semantic = F.normalize(self.visual_similarity(tokens), dim=-1)
        text_semantic = F.normalize(self.text_projection(text), dim=-1)
        similarity = torch.einsum("bnd,bd->bn", visual_semantic, text_semantic)
        similarity = similarity * self.logit_scale.clamp(max=4.6052).exp()
        visual = self.visual_decoder(tokens)
        gamma, beta = self.film(text_semantic).chunk(2, dim=-1)
        visual = visual * (1.0 + gamma[:, None]) + beta[:, None]
        visual = visual.transpose(1, 2).reshape(images.shape[0], -1, h, w)
        similarity = similarity.reshape(images.shape[0], 1, h, w)
        return self.decoder(
            torch.cat([visual, similarity], dim=1), images.shape[-2:]
        )


class ThreeHeadSegmentationBaseline(nn.Module):
    def __init__(
        self, backbone, *, visual_dim=768, decoder_dim=128, patch_size=16
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.Conv2d(visual_dim, decoder_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_dim, 3, 1),
        )

    def forward(self, images, conditions):
        tokens, (h, w) = self.backbone(images, return_spatial=True)
        features = tokens.transpose(1, 2).reshape(images.shape[0], -1, h, w)
        logits = F.interpolate(
            self.decoder(features),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        try:
            channels = torch.tensor(
                [TARGET_CLASS_TO_CHANNEL[int(value)] for value in conditions],
                device=logits.device,
            )
        except KeyError as exc:
            raise ValueError(f"unsupported prompt target class id: {exc.args[0]}") from exc
        rows = torch.arange(logits.shape[0], device=logits.device)
        return logits[rows, channels]
