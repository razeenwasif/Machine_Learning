"""Dataclasses and helpers for exploratory analysis and cleaning results."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ColumnProfile:
    """Lightweight profile describing a single column."""

    name: str
    dtype: str
    missing_count: int
    missing_fraction: float
    unique_values: int
    example_values: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["example_values"] = [self._coerce_value(value) for value in self.example_values]
        return data

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        return str(value)


@dataclass
class NumericProfile:
    """Extended statistics for numeric columns."""

    name: str
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    skew: float
    kurtosis: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CategoricalProfile:
    """Frequency table for categorical columns."""

    name: str
    top_frequencies: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "top_frequencies": [
                (str(category), int(count)) for category, count in self.top_frequencies
            ],
        }


@dataclass
class CorrelationResult:
    """Pair-wise correlation entry."""

    feature_a: str
    feature_b: str
    correlation: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetSummary:
    """Distribution metadata for the target column."""

    type: str
    distribution: Dict[str, float]
    imbalance_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["distribution"] = {str(k): float(v) for k, v in self.distribution.items()}
        if self.imbalance_ratio is not None:
            data["imbalance_ratio"] = float(self.imbalance_ratio)
        return data


@dataclass
class AnalysisReport:
    """Container aggregating EDA results for downstream consumption."""

    dataset_shape: Tuple[int, int]
    column_profiles: List[ColumnProfile]
    numeric_profiles: List[NumericProfile]
    categorical_profiles: List[CategoricalProfile]
    missing_by_column: Dict[str, float]
    correlation_pairs: List[CorrelationResult]
    duplicate_rows: int
    target_summary: Optional[TargetSummary]
    original_preview: List[Dict[str, Any]]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_shape": list(self.dataset_shape),
            "column_profiles": [profile.to_dict() for profile in self.column_profiles],
            "numeric_profiles": [profile.to_dict() for profile in self.numeric_profiles],
            "categorical_profiles": [profile.to_dict() for profile in self.categorical_profiles],
            "missing_by_column": {str(k): float(v) for k, v in self.missing_by_column.items()},
            "correlation_pairs": [pair.to_dict() for pair in self.correlation_pairs],
            "duplicate_rows": int(self.duplicate_rows),
            "target_summary": self.target_summary.to_dict() if self.target_summary else None,
            "original_preview": self._serialise_preview(self.original_preview),
            "notes": list(self.notes),
        }

    @staticmethod
    def _serialise_preview(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        serialised = []
        for row in rows:
            serialised.append({key: ColumnProfile._coerce_value(value) for key, value in row.items()})
        return serialised


@dataclass
class CleaningReport:
    """Record of the automatic cleaning steps that were applied."""

    applied_steps: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)
    filled_columns: Dict[str, str] = field(default_factory=dict)
    duplicate_rows_removed: int = 0
    outlier_treatments: Dict[str, str] = field(default_factory=dict)

    def has_changes(self) -> bool:
        return bool(
            self.applied_steps
            or self.dropped_columns
            or self.filled_columns
            or self.duplicate_rows_removed
            or self.outlier_treatments
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied_steps": list(self.applied_steps),
            "dropped_columns": list(self.dropped_columns),
            "filled_columns": {str(k): str(v) for k, v in self.filled_columns.items()},
            "duplicate_rows_removed": int(self.duplicate_rows_removed),
            "outlier_treatments": {str(k): str(v) for k, v in self.outlier_treatments.items()},
        }
