import torch
import torch.nn as nn
import torch.nn.functional as F
from geoadapter.adapters.base import ModalityAdapter
from geoadapter.models.prithvi import PrithviBackbone


class FocalLoss(nn.Module):
    """Multi-class focal loss for severely imbalanced semantic segmentation.

    Args:
        gamma: focusing parameter; γ=0 reduces to standard CE, γ=2 is the
            RetinaNet default that down-weights easy (well-classified) pixels
            so gradient is dominated by hard / minority-class pixels.
        alpha: optional per-class weight (list of length num_classes). For
            balanced binary seg you can leave it None.
        ignore_index: standard CE ignore_index passthrough.
    """

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha else None
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)  target: (B, H, W)
        log_p = F.log_softmax(logits, dim=1)
        if self.alpha is not None and self.alpha.device != logits.device:
            self.alpha = self.alpha.to(logits.device)
        ce = F.nll_loss(log_p, target, weight=self.alpha,
                        ignore_index=self.ignore_index, reduction="none")
        # pt = exp(-ce) is the prob of the *true* class
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


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
        class_weights: list[float] | None = None,
        loss: str = "ce",
        focal_gamma: float = 2.0,
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
            if loss == "focal":
                self.criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights,
                                           ignore_index=255)
            else:
                weight_t = torch.tensor(class_weights, device=device, dtype=torch.float32) \
                    if class_weights else None
                self.criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weight_t)
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
