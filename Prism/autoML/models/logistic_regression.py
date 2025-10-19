"""GPU-accelerated logistic regression."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .base import SupervisedTorchModel


class LogisticRegressionModel(SupervisedTorchModel):
    name = "logistic_regression"
    task_type = "classification"

    def __init__(self, device: torch.device, config: Optional[Dict] = None) -> None:
        super().__init__(device, config)
        self._is_binary = False

    def build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    def build_criterion(self, targets: torch.Tensor) -> nn.Module:
        n_classes = int(torch.unique(targets).numel())
        self._is_binary = n_classes <= 2
        if self._is_binary:
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        logits = super().predict(features)
        if logits.ndim == 1:
            preds = torch.sigmoid(logits)
            return (preds >= 0.5).long()
        return torch.argmax(logits, dim=1)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        logits = super().predict(features)
        if self._is_binary and logits.ndim == 1:
            probs = torch.sigmoid(logits)
            return torch.stack([1 - probs, probs], dim=1)
        return torch.softmax(logits, dim=1)
