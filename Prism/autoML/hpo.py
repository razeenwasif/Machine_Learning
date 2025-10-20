"""Optuna-powered hyperparameter optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import optuna
import torch

from .models.base import BaseModel
from .models.model_factory import Candidate, instantiate_model
from .utils.device import release_gpu_cache


SamplerFn = Callable[[optuna.trial.Trial, str, Dict], Dict]
MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _noop_sampler(trial: optuna.trial.Trial, model_key: str, context: Dict) -> Dict:
    del trial, model_key, context
    return {}


SEARCH_SPACES: Dict[str, SamplerFn] = {}


def register_search_space(model_name: str) -> Callable[[SamplerFn], SamplerFn]:
    def decorator(fn: SamplerFn) -> SamplerFn:
        SEARCH_SPACES[model_name] = fn
        return fn

    return decorator


@register_search_space("linear_regression")
def _linear_regression_sampler(trial: optuna.trial.Trial, model_key: str, context: Dict) -> Dict:
    prefix = f"{model_key}__"
    return {
        "lr": trial.suggest_float(prefix + "lr", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float(prefix + "weight_decay", 1e-8, 1e-3, log=True),
        "epochs": trial.suggest_int(prefix + "epochs", 50, 200),
        "batch_size": trial.suggest_categorical(prefix + "batch_size", [128, 256, 512, 1024]),
    }


@register_search_space("logistic_regression")
def _logistic_regression_sampler(trial: optuna.trial.Trial, model_key: str, context: Dict) -> Dict:
    prefix = f"{model_key}__"
    return {
        "lr": trial.suggest_float(prefix + "lr", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float(prefix + "weight_decay", 1e-8, 1e-3, log=True),
        "epochs": trial.suggest_int(prefix + "epochs", 80, 250),
        "batch_size": trial.suggest_categorical(prefix + "batch_size", [128, 256, 512, 1024]),
        "patience": trial.suggest_int(prefix + "patience", 3, 12),
    }


@register_search_space("neural_network")
def _neural_network_sampler(trial: optuna.trial.Trial, model_key: str, context: Dict) -> Dict:
    prefix = f"{model_key}__"
    n_hidden = trial.suggest_int(prefix + "num_hidden_layers", 1, 4)
    hidden_layers = []
    for idx in range(n_hidden):
        hidden_layers.append(trial.suggest_int(prefix + f"hidden_{idx}", 64, 512, log=True))
    config = {
        "lr": trial.suggest_float(prefix + "lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float(prefix + "weight_decay", 1e-8, 1e-3, log=True),
        "epochs": trial.suggest_int(prefix + "epochs", 80, 300),
        "batch_size": trial.suggest_categorical(prefix + "batch_size", [128, 256, 512, 1024]),
        "dropout": trial.suggest_float(prefix + "dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical(prefix + "activation", ["relu", "gelu", "leaky_relu"]),
        "hidden_layers": hidden_layers,
        "patience": trial.suggest_int(prefix + "patience", 5, 15),
    }
    if context.get("task") == "regression":
        config["task_type"] = "regression"
    else:
        config["task_type"] = "classification"
    return config


@register_search_space("kmeans")
def _kmeans_sampler(trial: optuna.trial.Trial, model_key: str, context: Dict) -> Dict:
    prefix = f"{model_key}__"
    n_samples = int(context.get("n_samples", 100))
    upper_bound = max(2, min(50, int(np.sqrt(n_samples))))
    lower_bound = min(upper_bound, 2)
    return {
        "n_clusters": trial.suggest_int(prefix + "n_clusters", lower_bound, upper_bound),
        "max_iter": trial.suggest_int(prefix + "max_iter", 50, 400),
        "tol": trial.suggest_float(prefix + "tol", 1e-5, 1e-3, log=True),
    }


@register_search_space("gaussian_mixture")
def _gaussian_mixture_sampler(trial: optuna.trial.Trial, model_key: str, context: Dict) -> Dict:
    prefix = f"{model_key}__"
    n_samples = int(context.get("n_samples", 100))
    upper_bound = max(2, min(20, n_samples // 5))
    lower_bound = min(upper_bound, 2)
    return {
        "n_components": trial.suggest_int(prefix + "n_components", lower_bound, upper_bound),
        "covariance_type": trial.suggest_categorical(
            prefix + "covariance_type",
            ["full", "tied", "diag", "spherical"],
        ),
        "reg_covar": trial.suggest_float(prefix + "reg_covar", 1e-8, 1e-3, log=True),
        "max_iter": trial.suggest_int(prefix + "max_iter", 100, 500),
        "init_params": trial.suggest_categorical(prefix + "init_params", ["kmeans", "random"]),
    }


@dataclass
class HPOResult:
    model_name: str
    config: Dict
    best_score: float
    study: optuna.Study


class HyperparameterOptimizer:
    """Wrapper around Optuna for model-aware search."""

    def __init__(
        self,
        task: str,
        device: torch.device,
        metric_fn: MetricFn,
        direction: str,
        n_trials: int = 20,
        seed: Optional[int] = None,
        context: Optional[Dict] = None,
    ) -> None:
        self.task = task
        self.device = device
        self.metric_fn = metric_fn
        self.direction = direction
        self.n_trials = n_trials
        self.seed = seed
        self.context = context or {}

    def optimize(
        self,
        candidates: Iterable[Tuple[str, Candidate]],
        train_data: Tuple[torch.Tensor, Optional[torch.Tensor]],
        val_data: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> HPOResult:
        train_features, train_targets = train_data
        val_features, val_targets = val_data

        candidate_map: Dict[str, Candidate] = {name: candidate for name, candidate in candidates}
        if not candidate_map:
            raise ValueError("No candidate models provided for optimisation.")

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            model_key = trial.suggest_categorical("model_name", list(candidate_map.keys()))
            model_cls, base_config = candidate_map[model_key]

            search_sampler = SEARCH_SPACES.get(model_key, _noop_sampler)
            sampled_config = search_sampler(trial, model_key, {**self.context, "task": self.task})
            merged_config = {**base_config, **sampled_config}

            model = instantiate_model(model_cls, self.device, merged_config)

            try:
                metric = self._train_and_evaluate(
                    model,
                    merged_config,
                    train_features,
                    train_targets,
                    val_features,
                    val_targets,
                )
            finally:
                model.cleanup()

            trial.set_user_attr("model_name", model_key)
            trial.set_user_attr("model_config", merged_config)
            release_gpu_cache()
            return metric

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_trial = study.best_trial
        best_name = best_trial.user_attrs["model_name"]
        best_config = best_trial.user_attrs["model_config"]
        best_score = best_trial.value
        return HPOResult(model_name=best_name, config=best_config, best_score=best_score, study=study)

    def _train_and_evaluate(
        self,
        model: BaseModel,
        config: Dict,
        train_features: torch.Tensor,
        train_targets: Optional[torch.Tensor],
        val_features: torch.Tensor,
        val_targets: Optional[torch.Tensor],
    ) -> float:
        if model.task_type != "clustering":
            if train_targets is None or val_targets is None:
                raise ValueError("Supervised tasks require targets for training and validation.")
            model.fit(train_features, train_targets, val_features, val_targets)
            predictions = model.predict(val_features).numpy()
            score = self.metric_fn(val_targets.numpy(), predictions)
            return float(score)

        # Clustering evaluation
        model.fit(train_features, None)
        predictions = model.predict(val_features).numpy()
        score = self.metric_fn(val_features.numpy(), predictions)
        return float(score)
