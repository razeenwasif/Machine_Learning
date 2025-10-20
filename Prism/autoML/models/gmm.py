"""Gaussian Mixture clustering model leveraging scikit-learn."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import numpy as np
from sklearn.mixture import GaussianMixture

from .base import BaseModel


class GaussianMixtureClusteringModel(BaseModel):
    """Wrapper around sklearn's GaussianMixture to fit the BaseModel API."""

    name = "gaussian_mixture"
    task_type = "clustering"

    def __init__(self, device: torch.device, config: Optional[Dict] = None) -> None:
        super().__init__(device, config)
        self.model: Optional[GaussianMixture] = None

    def fit(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        val_features: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> None:
        del targets, val_features, val_targets

        X = features.detach().cpu().numpy()
        if X.dtype != np.float64:
            X = X.astype(np.float64, copy=False)
        n_samples = X.shape[0]
        default_components = min(8, max(2, n_samples // 10))
        n_components = int(self.config.get("n_components", default_components))
        covariance_type = str(self.config.get("covariance_type", "full"))
        reg_covar = float(self.config.get("reg_covar", 1e-6))
        max_iter = int(self.config.get("max_iter", 300))
        init_params = str(self.config.get("init_params", "kmeans"))
        random_state = self.config.get("seed")

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=max_iter,
            init_params=init_params,
            random_state=random_state,
        )
        try:
            self.model.fit(X)
        except ValueError as exc:
            # Retry with a safer regularisation if covariance collapse is detected.
            if "ill-defined empirical covariance" in str(exc):
                boosted_reg = max(reg_covar, 1e-4)
                self.model.set_params(reg_covar=boosted_reg)
                self.model.fit(X)
            else:
                raise

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = features.detach().cpu().numpy()
        if X.dtype != np.float64:
            X = X.astype(np.float64, copy=False)
        labels = self.model.predict(X)
        return torch.from_numpy(labels)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        X = features.detach().cpu().numpy()
        if X.dtype != np.float64:
            X = X.astype(np.float64, copy=False)
        probs = self.model.predict_proba(X)
        return torch.from_numpy(probs)
