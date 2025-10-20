"""FastAPI backend for the Record Linkage pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path to allow sibling imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoML.linkage.pipeline import (
    RecordLinkagePipeline,
    RecordLinkageResult,
    RecordLinkageDependencyError,
)
from recordLinkage.src import PipelineConfigError

app = FastAPI()


class LinkageRequest(BaseModel):
    """Request model for the record linkage endpoint."""

    dataset_key: str = Field(..., description="The dataset preset key from the TOML config.")
    config_path: Optional[str] = Field(None, description="Optional path to a TOML configuration file.")
    output_path: Optional[str] = Field(None, description="Optional path to override the output CSV file.")
    use_gpu: Optional[bool] = Field(True, description="Whether to force GPU-accelerated comparisons.")
    skip_filters: bool = Field(False, description="Whether to skip precision filters.")


@app.post("/run", response_model=RecordLinkageResult)
def run_record_linkage(request: LinkageRequest) -> RecordLinkageResult:
    """
    Run the record linkage pipeline with the given configuration.
    """
    try:
        pipeline = RecordLinkagePipeline(
            config_path=Path(request.config_path) if request.config_path else None
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        output_path_arg = Path(request.output_path) if request.output_path else None
        result = pipeline.run(
            dataset_key=request.dataset_key,
            output_path=output_path_arg,
            use_gpu=request.use_gpu,
            skip_filters=request.skip_filters,
        )
        return result
    except (PipelineConfigError, RecordLinkageDependencyError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {exc}")


@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}
