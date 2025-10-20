"""GPU-enabled XGBoost models integrated with the AutoML pipeline."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import xgboost as xgb

from .base import BaseModel


def _default_tree_method(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return "gpu_hist"
    return "hist"


class _BaseXGBoostModel(BaseModel):
    """Shared XGBoost wrapper utilities."""

    def __init__(self, device: torch.device, config: Optional[Dict] = None) -> None:
        super().__init__(device, config)
        self.booster: Optional[xgb.Booster] = None
        self.best_ntree_limit: Optional[int] = None

    def _build_matrix(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> xgb.DMatrix:
        X = features.detach().cpu().numpy()
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        if targets is not None:
            y = targets.detach().cpu().numpy()
            return xgb.DMatrix(X, label=y)
        return xgb.DMatrix(X)

    def cleanup(self) -> None:
        self.booster = None
        self.best_ntree_limit = None
        super().cleanup()

    def _predict(self, features: torch.Tensor) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        dtest = self._build_matrix(features)
        kwargs: Dict = {}
        if self.best_ntree_limit and self.best_ntree_limit > 0:
            kwargs["iteration_range"] = (0, int(self.best_ntree_limit))
        return self.booster.predict(dtest, **kwargs)


class XGBoostRegressorModel(_BaseXGBoostModel):
    name = "xgboost_regressor"
    task_type = "regression"

    def fit(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        val_features: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> None:
        if targets is None:
            raise ValueError("Regression model expects targets.")

        use_gpu = self.device.type == "cuda" and torch.cuda.is_available()
        params = {
            "objective": "reg:squarederror",
            "tree_method": _default_tree_method(self.device),
            "predictor": "gpu_predictor" if use_gpu else "auto",
            "verbosity": 0,
        }
        params.update(self.config or {})

        num_boost_round = int(params.pop("num_boost_round", 400))
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))

        dtrain = self._build_matrix(features, targets)
        evals = [(dtrain, "train")]

        if val_features is not None and val_targets is not None:
            dval = self._build_matrix(val_features, val_targets)
            evals.append((dval, "validation"))
        else:
            dval = None

        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if len(evals) > 1 else None,
            verbose_eval=False,
        )
        if self.booster is None:
            raise RuntimeError("XGBoost training did not produce a booster.")
        best_limit = getattr(self.booster, "best_ntree_limit", 0) or self.booster.num_boosted_rounds()
        self.best_ntree_limit = best_limit

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        preds = self._predict(features)
        return torch.from_numpy(preds.astype(np.float32))


class XGBoostClassifierModel(_BaseXGBoostModel):
    name = "xgboost_classifier"
    task_type = "classification"

    def __init__(self, device: torch.device, config: Optional[Dict] = None) -> None:
        super().__init__(device, config)
        self.num_class: Optional[int] = None

    def fit(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        val_features: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> None:
        if targets is None:
            raise ValueError("Classification model expects targets.")

        y = targets.detach().cpu().numpy()
        if y.dtype != np.int64:
            y = y.astype(np.int64, copy=False)
        unique_classes = np.unique(y)
        self.num_class = int(unique_classes.max() + 1)

        is_binary = self.num_class <= 2
        use_gpu = self.device.type == "cuda" and torch.cuda.is_available()
        params = {
            "objective": "binary:logistic" if is_binary else "multi:softprob",
            "tree_method": _default_tree_method(self.device),
            "predictor": "gpu_predictor" if use_gpu else "auto",
            "verbosity": 0,
            "eval_metric": "logloss" if is_binary else "mlogloss",
        }
        if not is_binary:
            params["num_class"] = self.num_class
        params.update(self.config or {})

        num_boost_round = int(params.pop("num_boost_round", 400))
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))

        dtrain = self._build_matrix(features, targets)
        evals = [(dtrain, "train")]

        if val_features is not None and val_targets is not None:
            dval = self._build_matrix(val_features, val_targets)
            evals.append((dval, "validation"))

        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if len(evals) > 1 else None,
            verbose_eval=False,
        )
        if self.booster is None:
            raise RuntimeError("XGBoost training did not produce a booster.")
        best_limit = getattr(self.booster, "best_ntree_limit", 0) or self.booster.num_boosted_rounds()
        self.best_ntree_limit = best_limit

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        preds = self._predict(features)

        if self.num_class is None or self.num_class <= 2:
            labels = (preds > 0.5).astype(np.int64)
            return torch.from_numpy(labels)

        preds = preds.reshape(-1, self.num_class)
        labels = np.argmax(preds, axis=1).astype(np.int64)
        return torch.from_numpy(labels)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        preds = self._predict(features)
        if self.num_class is None or self.num_class <= 2:
            probs = np.vstack([1.0 - preds, preds]).T.astype(np.float32)
        else:
            probs = preds.astype(np.float32)
        return torch.from_numpy(probs)
