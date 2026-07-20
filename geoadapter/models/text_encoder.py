from __future__ import annotations

import torch
import torch.nn as nn


class SiglipTextEncoder(nn.Module):
    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        *,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoTokenizer, SiglipTextModel
        except ImportError as exc:
            raise ImportError(
                "Install GeoVLM dependencies with pip install -e '.[geovlm]'"
            ) from exc
        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.model = SiglipTextModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.output_dim = int(self.model.config.hidden_size)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, prompts: list[str] | tuple[str, ...]) -> torch.Tensor:
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must contain non-empty text")
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(list(prompts), padding=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.pooler_output
