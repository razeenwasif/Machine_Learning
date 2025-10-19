"""Flexible data loading for common structured formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd


class UnsupportedFormatError(RuntimeError):
    """Raised when a dataset format is not recognised."""


LoadFn = Callable[[Path], pd.DataFrame]


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as fh:
        first_char = fh.read(1)
        fh.seek(0)
        if first_char == "[":
            data = json.load(fh)
            return pd.DataFrame(data)
    # Assume JSON Lines by default
    return pd.read_json(path, lines=True)


LOADERS: Dict[str, LoadFn] = {
    ".csv": _load_csv,
    ".tsv": _load_csv,
    ".txt": _load_csv,
    ".parquet": _load_parquet,
    ".json": _load_json,
    ".jsonl": _load_json,
}


@dataclass
class DatasetBundle:
    """Container for loaded dataset and metadata."""

    frame: pd.DataFrame
    path: Path
    target: Optional[str] = None


def load_dataset(path: str, target: Optional[str] = None) -> DatasetBundle:
    """Load a dataset from disk inferring the appropriate loader."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    loader = LOADERS.get(resolved.suffix.lower())
    if loader is None:
        raise UnsupportedFormatError(f"Unsupported file extension: {resolved.suffix}")

    frame = loader(resolved)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Loader did not return a pandas DataFrame.")

    if frame.empty:
        raise ValueError("Loaded dataset is empty.")

    return DatasetBundle(frame=frame, path=resolved, target=target)
