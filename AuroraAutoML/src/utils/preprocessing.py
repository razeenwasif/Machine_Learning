"""Preprocessing utilities for heterogeneous tabular data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api import types as pdt


@dataclass
class Preprocessor:
    """Fit/transform utility that converts pandas DataFrames to torch tensors."""

    target_column: Optional[str] = None
    task_type: Optional[str] = None
    transformer: Optional[ColumnTransformer] = field(default=None, init=False)
    feature_names_: List[str] = field(default_factory=list, init=False)
    target_mapping_: Dict[Any, int] = field(default_factory=dict, init=False)
    inverse_target_mapping_: Dict[int, Any] = field(default_factory=dict, init=False)

    def _split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self.target_column and self.target_column in df.columns:
            y = df[self.target_column]
            X = df.drop(columns=[self.target_column])
        else:
            X, y = df, None
        return X, y

    def _build_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []

        if numeric_cols:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", numeric_pipeline, numeric_cols))

        if categorical_cols:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", categorical_pipeline, categorical_cols))

        if not transformers:
            raise ValueError("No usable columns found for preprocessing.")

        return ColumnTransformer(transformers=transformers)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        X_df, y_series = self._split_features_target(df)
        self.transformer = self._build_transformer(X_df)
        X_np = self.transformer.fit_transform(X_df)

        # Track generated feature names for logging/debugging
        feature_names = []
        for name, trans, cols in self.transformer.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                encoder: OneHotEncoder = trans.named_steps["encoder"]  # type: ignore[assignment]
                feature_names.extend(encoder.get_feature_names_out(cols))
        self.feature_names_ = feature_names

        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = None
        if y_series is not None:
            y_tensor = self._prepare_target(y_series, fit=True)
        return X_tensor, y_tensor

    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.transformer is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        X_df, y_series = self._split_features_target(df)
        X_np = self.transformer.transform(X_df)
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = None
        if y_series is not None:
            y_tensor = self._prepare_target(y_series, fit=False)
        return X_tensor, y_tensor

    def _prepare_target(self, series: pd.Series, fit: bool) -> torch.Tensor:
        if self.task_type == "regression":
            return torch.tensor(series.to_numpy(), dtype=torch.float32)

        if self.task_type == "classification" or (self.task_type is None and not pdt.is_numeric_dtype(series)):
            if fit:
                categories = series.astype("category")
                self.target_mapping_ = {cat: idx for idx, cat in enumerate(categories.cat.categories)}
                self.inverse_target_mapping_ = {idx: cat for cat, idx in self.target_mapping_.items()}
                codes = categories.cat.codes.to_numpy()
            else:
                if not self.target_mapping_:
                    raise RuntimeError("Target encoder has not been fitted.")
                codes = series.map(self.target_mapping_)
                if codes.isnull().any():
                    raise ValueError("Encountered unseen category in target column during transform().")
                codes = codes.to_numpy()
            return torch.tensor(codes, dtype=torch.long)

        # Default behaviour: numeric targets -> choose dtype based on detected kind
        if pdt.is_integer_dtype(series):
            if self.task_type == "regression":
                return torch.tensor(series.to_numpy(), dtype=torch.float32)
            return torch.tensor(series.to_numpy(), dtype=torch.long)
        return torch.tensor(series.to_numpy(), dtype=torch.float32)
