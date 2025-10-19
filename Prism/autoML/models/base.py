"""Base classes for torch-based GPU models."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.device import release_gpu_cache


class BaseModel(ABC):
    """Abstract model wrapper."""

    name: str
    task_type: str

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> None:
        self.device = device
        self.config = config or {}

    @abstractmethod
    def fit(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        val_features: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> None:
        ...

    @abstractmethod
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        ...

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release GPU memory."""
        for attr in list(self.__dict__.keys()):
            if isinstance(getattr(self, attr), torch.nn.Module):
                delattr(self, attr)
        release_gpu_cache()


class SupervisedTorchModel(BaseModel):
    """Torch helper that implements a standard training loop."""

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(device, config=config)
        self.model: Optional[nn.Module] = None

    def build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        raise NotImplementedError

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        lr = self.config.get("lr", 1e-3)
        weight_decay = self.config.get("weight_decay", 0.0)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def build_criterion(self, targets: torch.Tensor) -> nn.Module:
        raise NotImplementedError

    def _prepare_data_loader(self, features: torch.Tensor, targets: torch.Tensor) -> DataLoader:
        batch_size = self.config.get("batch_size", 256)
        dataset = TensorDataset(features, targets)
        pin = torch.cuda.is_available()
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)

    def _training_step(
        self,
        model: nn.Module,
        batch_features: torch.Tensor,
        batch_targets: torch.Tensor,
        criterion: nn.Module,
        scaler: Optional[Any],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = torch.cuda.is_available()
        if autocast_enabled:
            try:
                autocast_ctx = torch.amp.autocast("cuda")
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast()
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return float(loss.detach().cpu())

    def _eval_step(
        self,
        model: nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        model.eval()
        with torch.no_grad():
            outputs = model(features)
            loss = criterion(outputs, targets)
        return float(loss.detach().cpu())

    def fit(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        val_features: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> None:
        if targets is None:
            raise ValueError("Supervised model expects targets.")

        features_cpu = features.detach().cpu()
        targets_cpu = targets.detach().cpu()
        input_dim = features_cpu.shape[1]
        output_dim = self._infer_output_dim(targets)
        self.model = self.build_model(input_dim, output_dim).to(self.device)
        optimizer = self.build_optimizer(self.model)
        criterion = self.build_criterion(targets)
        scaler = self._create_grad_scaler()

        epochs = self.config.get("epochs", 30)
        patience = self.config.get("patience", 5)

        data_loader = self._prepare_data_loader(features_cpu, targets_cpu)
        val_features_device = None
        val_targets_device = None
        if val_features is not None and val_targets is not None:
            val_features_device = val_features.to(self.device, non_blocking=True)
            val_targets_device = val_targets.to(self.device, non_blocking=True)
        best_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        for _ in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                running_loss += self._training_step(
                    self.model, batch_features, batch_targets, criterion, scaler, optimizer
                )
            avg_loss = running_loss / max(len(data_loader), 1)

            val_loss = avg_loss
            if val_features_device is not None and val_targets_device is not None:
                val_loss = self._eval_step(self.model, val_features_device, val_targets_device, criterion)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def _create_grad_scaler(self) -> Optional[Any]:
        if not torch.cuda.is_available():
            return None
        try:
            from torch.amp import GradScaler as AmpGradScaler  # type: ignore[attr-defined]

            return AmpGradScaler(device_type="cuda")
        except (ImportError, TypeError):
            return torch.cuda.amp.GradScaler()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        self.model.eval()
        features = features.to(self.device, non_blocking=True)
        with torch.no_grad():
            outputs = self.model(features)
        return outputs.detach().cpu()

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        outputs = self.predict(features)
        return torch.softmax(outputs, dim=1)

    def _infer_output_dim(self, targets: torch.Tensor) -> int:
        if targets.dtype in (torch.long, torch.int64):
            n_classes = int(torch.unique(targets).numel())
            return max(n_classes, 2)
        return 1
