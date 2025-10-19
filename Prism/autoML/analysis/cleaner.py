"""Automatic data cleaning helpers building upon the analysis report."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .report import AnalysisReport, CleaningReport


class DataCleaner:
    """Apply conservative cleaning steps guided by an analysis report."""

    def __init__(
        self,
        *,
        missing_threshold: float = 0.6,
        enable_outlier_clipping: bool = True,
    ) -> None:
        if not 0 <= missing_threshold <= 1:
            raise ValueError("missing_threshold must be between 0 and 1.")
        self.missing_threshold = missing_threshold
        self.enable_outlier_clipping = enable_outlier_clipping

    def clean(self, frame: pd.DataFrame, analysis: AnalysisReport) -> Tuple[pd.DataFrame, CleaningReport]:
        cleaned = frame.copy()
        report = CleaningReport()

        # Drop duplicate rows up-front
        duplicates = int(cleaned.duplicated().sum())
        if duplicates:
            cleaned = cleaned.drop_duplicates()
            report.duplicate_rows_removed = duplicates
            report.applied_steps.append(f"Removed {duplicates} duplicate rows.")

        # Drop columns that are mostly empty
        columns_to_drop = [
            column
            for column, fraction in analysis.missing_by_column.items()
            if fraction >= self.missing_threshold and column in cleaned.columns
        ]
        if columns_to_drop:
            cleaned = cleaned.drop(columns=columns_to_drop)
            report.dropped_columns.extend(columns_to_drop)
            report.applied_steps.append(
                f"Dropped {len(columns_to_drop)} columns with ≥{int(self.missing_threshold * 100)}% missing values."
            )

        # Fill remaining missing values using simple, robust strategies
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric_cols:
            if cleaned[column].isna().any():
                median = cleaned[column].median()
                cleaned[column] = cleaned[column].fillna(median)
                report.filled_columns[column] = "median"
        categorical_cols = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
        for column in categorical_cols:
            if cleaned[column].isna().any():
                mode = cleaned[column].mode(dropna=True)
                if not mode.empty:
                    fill_value = mode.iloc[0]
                else:
                    fill_value = "missing"
                cleaned[column] = cleaned[column].fillna(fill_value)
                report.filled_columns[column] = "mode"

        # Clip extreme outliers using the IQR rule
        if self.enable_outlier_clipping:
            outlier_columns = {}
            for column in numeric_cols:
                series = cleaned[column].dropna()
                if series.empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                mask = (cleaned[column] < lower_bound) | (cleaned[column] > upper_bound)
                if mask.any():
                    cleaned[column] = cleaned[column].clip(lower=lower_bound, upper=upper_bound)
                    outlier_columns[column] = f"clipped to [{lower_bound:.3g}, {upper_bound:.3g}]"
            if outlier_columns:
                report.outlier_treatments.update(outlier_columns)
                report.applied_steps.append(
                    f"Clipped outliers in {len(outlier_columns)} numeric columns using IQR bounds."
                )

        return cleaned, report
