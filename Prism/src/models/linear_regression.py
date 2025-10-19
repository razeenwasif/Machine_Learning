"""GPU-accelerated linear regression."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .base import SupervisedTorchModel


class LinearRegressionModel(SupervisedTorchModel):
    name = "linear_regression"
    task_type = "regression"

    def __init__(self, device: torch.device, config: Optional[Dict] = None) -> None:
        super().__init__(device, config)

    def build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    def build_criterion(self, targets: torch.Tensor) -> nn.Module:
        return nn.MSELoss()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        outputs = super().predict(features)
        return outputs.squeeze(-1)
