"""FastAPI backend for the AutoML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path to allow sibling imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoML.data.loaders import load_dataset
from autoML.pipeline import AutoMLPipeline, PipelineResult

app = FastAPI()


class PreviewRequest(BaseModel):
    """Request model for the dataset preview endpoint."""
    data_path: str = Field(..., description="Path to the dataset file.")


class AutoMLRequest(BaseModel):
    """Request model for the AutoML endpoint."""

    data_path: str = Field(..., description="Path to the dataset file.")
    target: Optional[str] = Field(None, description="Name of the target column for supervised learning.")
    task: str = Field("auto", description="Task type override (auto, regression, classification, clustering).")
    test_size: float = Field(0.2, description="Hold-out ratio for final evaluation.")
    max_trials: int = Field(20, description="Optuna trial budget.")
    seed: int = Field(42, description="Global random seed.")
    deterministic: bool = Field(False, description="Enforce deterministic CUDA kernels.")
    prefer_gpu: bool = Field(True, description="Prefer GPU execution when available.")


@app.post("/preview-dataset")
def preview_dataset(request: PreviewRequest):
    """
    Load the first 100 rows of a dataset and return it as JSON.
    """
    try:
        df = load_dataset(request.data_path).frame
        return df.head(100).to_dict(orient='records')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found at path: {request.data_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset preview: {e}")


@app.post("/run", response_model=PipelineResult)
def run_automl(request: AutoMLRequest) -> PipelineResult:
    """
    Run the AutoML pipeline with the given configuration.
    """
    try:
        pipeline = AutoMLPipeline(
            seed=request.seed,
            deterministic=request.deterministic,
            max_trials=request.max_trials,
            prefer_gpu=request.prefer_gpu,
        )
        result = pipeline.run(
            data_path=request.data_path,
            target=request.target,
            task=request.task,
            test_size=request.test_size,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}")


@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}
