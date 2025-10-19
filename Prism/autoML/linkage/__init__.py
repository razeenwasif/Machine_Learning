"""Integrated record linkage entry points for Prism."""

from .pipeline import RecordLinkageDependencyError, RecordLinkagePipeline, RecordLinkageResult

__all__ = ["RecordLinkageDependencyError", "RecordLinkagePipeline", "RecordLinkageResult"]
