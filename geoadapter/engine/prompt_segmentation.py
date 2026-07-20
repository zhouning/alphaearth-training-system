from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptSegmentationLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, epsilon=1e-6):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.epsilon = float(epsilon)

    def forward(self, logits, targets, positive_weights):
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have identical shapes")
        if positive_weights.ndim != 1 or positive_weights.shape[0] != logits.shape[0]:
            raise ValueError("positive_weights must have shape [B]")
        targets = targets.to(dtype=logits.dtype)
        pixel_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        weights = torch.where(
            targets > 0.5,
            positive_weights.to(logits.device)[:, None, None],
            torch.ones_like(targets),
        )
        bce = (pixel_loss * weights).mean()
        probabilities = logits.sigmoid().flatten(1)
        target_flat = targets.flatten(1)
        intersection = (probabilities * target_flat).sum(dim=1)
        dice = (2 * intersection + self.epsilon) / (
            probabilities.sum(dim=1) + target_flat.sum(dim=1) + self.epsilon
        )
        return self.bce_weight * bce + self.dice_weight * (1 - dice).mean()


class PromptSegmentationTrainer:
    def __init__(
        self,
        model,
        *,
        lr=1e-3,
        lr_peft=1e-4,
        epochs=50,
        device="cpu",
        loss=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = loss or PromptSegmentationLoss()
        prompt_params, peft_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = peft_params if "houlsby_adapter" in name else prompt_params
            target.append(parameter)
        groups = [{"params": prompt_params, "lr": lr}]
        if peft_params:
            groups.append({"params": peft_params, "lr": lr_peft})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def train_step(self, images, conditions, targets, positive_weights):
        self.model.train()
        images = images.to(self.device)
        targets = targets.to(self.device)
        positive_weights = positive_weights.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(images, conditions)
        loss = self.criterion(logits, targets, positive_weights)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, images, conditions):
        self.model.eval()
        return self.model(images.to(self.device), conditions)

    def state_dict(self, *, epoch, metadata):
        return {
            "epoch": int(epoch),
            "trainable_model": {
                name: parameter.detach().cpu()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metadata": dict(metadata),
        }

    def load_state_dict(self, state):
        missing, unexpected = self.model.load_state_dict(
            state["trainable_model"], strict=False
        )
        trainable_names = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        missing_trainable = sorted(set(missing) & trainable_names)
        if missing_trainable:
            raise ValueError(
                "missing trainable checkpoint parameters: "
                + ", ".join(missing_trainable)
            )
        if unexpected:
            raise ValueError(f"unexpected checkpoint parameters: {unexpected}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        return int(state["epoch"]), dict(state["metadata"])
