"""Factory utilities for model instantiation and search spaces."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Type

import torch

from .base import BaseModel
from .clustering import KMeansClusteringModel
from .linear_regression import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .neural_network import NeuralNetworkModel
from .gmm import GaussianMixtureClusteringModel
from .xgboost_model import XGBoostRegressorModel, XGBoostClassifierModel


Candidate = Tuple[Type[BaseModel], Dict]


def candidates_for_task(task: str) -> Iterable[Candidate]:
    if task == "regression":
        yield LinearRegressionModel, {"epochs": 60, "batch_size": 512, "lr": 1e-2}
        yield NeuralNetworkModel, {
            "task_type": "regression",
            "epochs": 80,
            "batch_size": 512,
            "lr": 5e-3,
            "hidden_layers": [256, 128],
            "dropout": 0.0,
        }
        yield XGBoostRegressorModel, {
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "lambda": 1.0,
            "alpha": 0.0,
            "eta": 0.1,
            "num_boost_round": 400,
            "early_stopping_rounds": 40,
        }
    elif task == "classification":
        yield LogisticRegressionModel, {"epochs": 80, "batch_size": 512, "lr": 5e-3}
        yield NeuralNetworkModel, {
            "task_type": "classification",
            "epochs": 100,
            "batch_size": 512,
            "lr": 1e-3,
            "hidden_layers": [512, 256],
            "dropout": 0.2,
        }
        yield XGBoostClassifierModel, {
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "lambda": 1.0,
            "alpha": 0.0,
            "eta": 0.1,
            "num_boost_round": 400,
            "early_stopping_rounds": 40,
        }
    elif task == "clustering":
        yield KMeansClusteringModel, {"max_iter": 200}
        yield GaussianMixtureClusteringModel, {
            "n_components": 4,
            "covariance_type": "full",
            "max_iter": 300,
            "reg_covar": 1e-6,
        }
    else:
        raise ValueError(f"Unknown task: {task}")


def instantiate_model(model_cls: Type[BaseModel], device: torch.device, config: Dict) -> BaseModel:
    return model_cls(device=device, config=config)
