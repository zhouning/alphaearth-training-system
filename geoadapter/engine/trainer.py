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
        lr_peft: float | None = None,
        epochs: int = 50,
        task: str = "classification",
        device: str = "cpu",
    ):
        self.backbone = backbone.to(device)
        self.adapter = adapter.to(device) if adapter else None
        self.head = head.to(device)
        self.device = device
        self.task = task
        self.return_spatial = (task == "segmentation")

        # Differential learning rates
        head_params = list(head.parameters())
        adapter_params = [p for p in (adapter.parameters() if adapter else []) if p.requires_grad]
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]

        param_groups = [{"params": head_params, "lr": lr}]
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": lr})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_peft or lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Task-specific loss
        if task == "multilabel":
            self.criterion = nn.BCEWithLogitsLoss()
        elif task == "segmentation":
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if self.adapter:
            x = self.adapter(x)
        if self.return_spatial:
            features, spatial_dims = self.backbone(x, return_spatial=True)
            logits = self.head(features, spatial_dims)
        else:
            features = self.backbone(x)
            logits = self.head(features)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def step_scheduler(self):
        """Call at end of each epoch."""
        self.scheduler.step()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.adapter:
            x = self.adapter(x)
        if self.return_spatial:
            features, spatial_dims = self.backbone(x, return_spatial=True)
            return self.head(features, spatial_dims)
        features = self.backbone(x)
        return self.head(features)
