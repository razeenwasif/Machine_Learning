"""Metric helpers that adapt to the selected task."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn import metrics

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def regression_metrics() -> Dict[str, MetricFn]:
    return {
        "rmse": lambda y_true, y_pred: float(
            np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        ),
        "mae": lambda y_true, y_pred: float(metrics.mean_absolute_error(y_true, y_pred)),
        "r2": lambda y_true, y_pred: float(metrics.r2_score(y_true, y_pred)),
    }


def classification_metrics() -> Dict[str, MetricFn]:
    return {
        "accuracy": lambda y_true, y_pred: float(metrics.accuracy_score(y_true, y_pred)),
        "f1_macro": lambda y_true, y_pred: float(metrics.f1_score(y_true, y_pred, average="macro")),
    }


def clustering_metrics(features: np.ndarray) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    # Clustering metrics often require the original feature space.
    return {
        "silhouette": lambda feats, labels: float(
            metrics.silhouette_score(feats, labels) if len(np.unique(labels)) > 1 else -1.0
        ),
        "calinski_harabasz": lambda feats, labels: float(
            metrics.calinski_harabasz_score(feats, labels) if len(np.unique(labels)) > 1 else -1.0
        ),
    }


def pick_primary_metric(task: str) -> Tuple[str, MetricFn]:
    if task == "regression":
        metrics_map = regression_metrics()
        return "rmse", metrics_map["rmse"]
    if task == "classification":
        metrics_map = classification_metrics()
        return "accuracy", metrics_map["accuracy"]
    if task == "clustering":
        def silhouette_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(metrics.silhouette_score(y_true, y_pred))  # type: ignore[arg-type]

        return "silhouette", silhouette_safe
    raise ValueError(f"Unknown task: {task}")
