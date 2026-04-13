import torch
import torch.nn as nn
from geoadapter.adapters.base import ModalityAdapter
from geoadapter.models.prithvi import PrithviBackbone


class PEFTTrainer:
    """Unified training loop for all PEFT methods."""

    def __init__(
        self,
        backbone: PrithviBackbone,
        adapter: ModalityAdapter | None,
        head: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.backbone = backbone.to(device)
        self.adapter = adapter.to(device) if adapter else None
        self.head = head.to(device)
        self.device = device

        params = list(head.parameters())
        if adapter:
            params += [p for p in adapter.parameters() if p.requires_grad]
        params += [p for p in backbone.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if self.adapter:
            x = self.adapter(x)
        features = self.backbone(x)
        logits = self.head(features)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.adapter:
            x = self.adapter(x)
        features = self.backbone(x)
        return self.head(features)
