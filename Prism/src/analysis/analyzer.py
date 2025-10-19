"""Exploratory data analysis utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .report import (
    AnalysisReport,
    CategoricalProfile,
    ColumnProfile,
    CorrelationResult,
    NumericProfile,
    TargetSummary,
)


class DataAnalyzer:
    """Produce structured exploratory analysis summaries for a dataset."""

    def __init__(self, max_correlation_pairs: int = 25) -> None:
        self.max_correlation_pairs = max_correlation_pairs

    def analyze(self, frame: pd.DataFrame, target: Optional[str] = None) -> AnalysisReport:
        numeric_profiles = []
        categorical_profiles = []
        column_profiles = []

        missing_by_column = (
            frame.isna().sum().divide(len(frame)).fillna(0.0).to_dict()
            if len(frame) else {}
        )
        duplicate_rows = int(frame.duplicated().sum())

        for column in frame.columns:
            series = frame[column]
            missing_count = int(series.isna().sum())
            missing_fraction = float(missing_by_column.get(column, 0.0))
            unique_values = int(series.nunique(dropna=True))
            example_values = series.dropna().unique().tolist()[:5]
            column_profiles.append(
                ColumnProfile(
                    name=column,
                    dtype=str(series.dtype),
                    missing_count=missing_count,
                    missing_fraction=missing_fraction,
                    unique_values=unique_values,
                    example_values=example_values,
                )
            )

            if pd.api.types.is_numeric_dtype(series):
                cleaned = series.dropna()
                if cleaned.empty:
                    continue
                desc = cleaned.describe(percentiles=[0.25, 0.5, 0.75])
                numeric_profiles.append(
                    NumericProfile(
                        name=column,
                        mean=float(desc.get("mean", np.nan)),
                        std=float(desc.get("std", np.nan)),
                        min=float(desc.get("min", np.nan)),
                        q1=float(desc.get("25%", np.nan)),
                        median=float(desc.get("50%", np.nan)),
                        q3=float(desc.get("75%", np.nan)),
                        max=float(desc.get("max", np.nan)),
                        skew=float(cleaned.skew()) if len(cleaned) > 2 else float("nan"),
                        kurtosis=float(cleaned.kurtosis()) if len(cleaned) > 3 else float("nan"),
                    )
                )
            else:
                value_counts = series.value_counts(dropna=True).head(5)
                categorical_profiles.append(
                    CategoricalProfile(
                        name=column,
                        top_frequencies=[(str(idx), int(count)) for idx, count in value_counts.items()],
                    )
                )

        correlation_pairs = self._compute_correlations(frame)
        target_summary = self._summarise_target(frame, target) if target else None
        original_preview = frame.head(20).to_dict(orient="records")

        notes = []
        if duplicate_rows:
            notes.append(f"Detected {duplicate_rows} duplicate rows.")
        high_missing = [col for col, frac in missing_by_column.items() if frac > 0.2]
        if high_missing:
            notes.append(f"Columns with >20% missing values: {', '.join(high_missing[:10])}")

        return AnalysisReport(
            dataset_shape=(len(frame), frame.shape[1]),
            column_profiles=column_profiles,
            numeric_profiles=numeric_profiles,
            categorical_profiles=categorical_profiles,
            missing_by_column={k: float(v) for k, v in missing_by_column.items()},
            correlation_pairs=correlation_pairs,
            duplicate_rows=duplicate_rows,
            target_summary=target_summary,
            original_preview=original_preview,
            notes=notes,
        )

    def _compute_correlations(self, frame: pd.DataFrame) -> list[CorrelationResult]:
        numeric_cols = frame.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return []
        corr_matrix = frame[numeric_cols].corr()
        corr_pairs = []
        for i, col_a in enumerate(corr_matrix.columns):
            for j, col_b in enumerate(corr_matrix.columns):
                if j <= i:
                    continue
                value = float(corr_matrix.loc[col_a, col_b])
                if np.isnan(value):
                    continue
                corr_pairs.append(CorrelationResult(feature_a=col_a, feature_b=col_b, correlation=value))
        corr_pairs.sort(key=lambda item: abs(item.correlation), reverse=True)
        return corr_pairs[: self.max_correlation_pairs]

    def _summarise_target(self, frame: pd.DataFrame, target: str) -> Optional[TargetSummary]:
        if target not in frame.columns:
            return None
        series = frame[target]
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            distribution = {
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
                "min": float(desc.get("min", np.nan)),
                "25%": float(desc.get("25%", np.nan)),
                "50%": float(desc.get("50%", np.nan)),
                "75%": float(desc.get("75%", np.nan)),
                "max": float(desc.get("max", np.nan)),
            }
            return TargetSummary(type="numeric", distribution=distribution, imbalance_ratio=None)

        value_counts = series.value_counts(dropna=False)
        total = value_counts.sum()
        distribution = {str(idx): float(count / total) for idx, count in value_counts.items()}
        if not distribution:
            return TargetSummary(type="categorical", distribution={}, imbalance_ratio=None)
        sorted_counts = value_counts.sort_values(ascending=False)
        if len(sorted_counts) > 1:
            imbalance_ratio = float(sorted_counts.iloc[0] / sorted_counts.iloc[1])
        else:
            imbalance_ratio = None
        return TargetSummary(type="categorical", distribution=distribution, imbalance_ratio=imbalance_ratio)
