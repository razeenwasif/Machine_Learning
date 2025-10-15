"""Configurable feedforward neural network."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from .base import SupervisedTorchModel


class NeuralNetworkModel(SupervisedTorchModel):
    name = "neural_network"
    task_type = "classification"

    def __init__(self, device: torch.device, config: Optional[Dict] = None) -> None:
        super().__init__(device, config)
        if config is None:
            config = {}
        self.task_type = config.get("task_type", "classification")

    def build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        hidden_layers: List[int] = self.config.get("hidden_layers", [256, 128])
        activation_name = self.config.get("activation", "relu").lower()
        dropout = float(self.config.get("dropout", 0.1))

        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        activation = activations.get(activation_name, nn.ReLU())

        layers: List[nn.Module] = []
        in_features = input_dim
        for hidden in hidden_layers:
            layers.extend([nn.Linear(in_features, hidden), activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden

        layers.append(nn.Linear(in_features, output_dim))
        if self.task_type == "regression":
            layers.append(nn.Identity())

        return nn.Sequential(*layers)

    def build_criterion(self, targets: torch.Tensor) -> nn.Module:
        if self.task_type == "regression":
            return nn.MSELoss()
        return nn.CrossEntropyLoss()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        outputs = super().predict(features)
        if self.task_type == "regression":
            return outputs.squeeze(-1)
        return torch.argmax(outputs, dim=1)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        if self.task_type == "regression":
            raise RuntimeError("Regression model does not produce probabilities.")
        logits = super().predict(features)
        return torch.softmax(logits, dim=1)
