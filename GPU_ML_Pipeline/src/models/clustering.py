"""GPU-accelerated clustering models."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .base import BaseModel


class KMeansClusteringModel(BaseModel):
    name = "kmeans"
    task_type = "clustering"

    def __init__(self, device: torch.device, config: Optional[Dict] = None) -> None:
        super().__init__(device, config)
        self.centroids: Optional[torch.Tensor] = None

    def fit(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        val_features: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> None:
        del targets, val_features, val_targets

        X = features.to(self.device, non_blocking=True)
        n_samples = X.shape[0]
        n_clusters = int(self.config.get("n_clusters", min(8, max(2, n_samples // 10))))
        max_iter = int(self.config.get("max_iter", 100))
        tol = float(self.config.get("tol", 1e-4))
        seed = self.config.get("seed")
        if seed is not None:
            torch.manual_seed(seed)

        # Init centroids via k-means++
        indices = [torch.randint(0, n_samples, (1,), device=self.device)]
        for _ in range(1, n_clusters):
            current_centroids = X[indices]
            distances = torch.cdist(X, current_centroids, p=2) ** 2
            min_dist, _ = torch.min(distances, dim=1)
            probs = min_dist / torch.sum(min_dist)
            next_idx = torch.multinomial(probs, 1)
            indices.append(next_idx)
        centroids = X[torch.cat(indices)]

        for _ in range(max_iter):
            distances = torch.cdist(X, centroids, p=2)
            labels = torch.argmin(distances, dim=1)

            new_centroids = []
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if torch.any(mask):
                    new_centroids.append(X[mask].mean(dim=0))
                else:
                    # Reinitialize empty cluster to random point
                    new_centroids.append(X[torch.randint(0, n_samples, (1,), device=self.device)].squeeze(0))
            new_centroids_tensor = torch.stack(new_centroids)

            shift = torch.norm(new_centroids_tensor - centroids, p=2)
            centroids = new_centroids_tensor
            if shift < tol:
                break

        self.centroids = centroids.detach().cpu()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = features.to(self.device, non_blocking=True)
        centroids = self.centroids.to(self.device)
        distances = torch.cdist(X, centroids, p=2)
        labels = torch.argmin(distances, dim=1)
        return labels.detach().cpu()

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        X = features.to(self.device, non_blocking=True)
        centroids = self.centroids.to(self.device)
        distances = torch.cdist(X, centroids, p=2)
        inv_dist = torch.exp(-distances)
        probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
        return probs.detach().cpu()
