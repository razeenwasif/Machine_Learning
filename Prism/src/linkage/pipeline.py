"""Prism integration layer for the GPU-powered record linkage pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..analysis.analyzer import DataAnalyzer
from ..analysis.report import AnalysisReport
from ..data.loaders import load_dataset
from recordLinkage.src import PipelineConfig, PipelineConfigError, load_pipeline_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "recordLinkage" / "config" / "pipeline.toml"


class RecordLinkageDependencyError(RuntimeError):
    """Raised when GPU dependencies required for record linkage are unavailable."""


@dataclass
class RecordLinkageResult:
    dataset_key: str
    dataset_a_path: Path
    dataset_b_path: Path
    truth_path: Optional[Path]
    output_path: Path
    id_column: str
    attributes: list[str]
    match_count: int
    non_match_count: int
    true_match_count: int
    candidate_pairs: int
    filtered_pairs: int
    blocking_metrics: Dict[str, float]
    linkage_metrics: Dict[str, float]
    runtime: Dict[str, float]
    analysis_a: AnalysisReport
    analysis_b: AnalysisReport


def _ensure_gpu_dependencies() -> None:
    """Validate that the RAPIDS GPU stack is installed before proceeding."""
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - requires GPU runtime
        missing = getattr(exc, "name", "required GPU libraries")
        raise RecordLinkageDependencyError(
            "Record linkage requires RAPIDS GPU dependencies "
            f"(missing module: {missing}). See recordLinkage/README.md for setup steps."
        ) from exc
    except Exception as exc:  # pragma: no cover - requires GPU runtime
        raise RecordLinkageDependencyError(
            f"Record linkage requires a functional CUDA environment: {exc}"
        ) from exc


class RecordLinkagePipeline:
    """Drive the GPU-based record linkage workflow inside Prism."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        resolved = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config_path = resolved.resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Record linkage configuration file not found: {self.config_path}")

    def run(
        self,
        dataset_key: Optional[str] = None,
        *,
        output_path: Optional[Path] = None,
        use_gpu: Optional[bool] = None,
        skip_filters: bool = False,
    ) -> RecordLinkageResult:
        """Execute the linkage workflow and return structured results."""
        _ensure_gpu_dependencies()

        from recordLinkage.src import recordLinkage as rl_cli  # Local import to avoid eager GPU dependency checks

        try:
            pipeline_cfg = load_pipeline_config(str(self.config_path), dataset_key=dataset_key)
        except PipelineConfigError as exc:
            raise PipelineConfigError(f"{exc}") from exc

        pipeline_cfg = self._apply_overrides(
            pipeline_cfg,
            output_path=output_path,
            use_gpu=use_gpu,
            skip_filters=skip_filters,
        )

        # Run Prism's analysis over both datasets for consistent reporting.
        analyzer = DataAnalyzer()
        dataset_a_bundle = load_dataset(pipeline_cfg.dataset_a)
        dataset_b_bundle = load_dataset(pipeline_cfg.dataset_b)
        analysis_a = analyzer.analyze(dataset_a_bundle.frame)
        analysis_b = analyzer.analyze(dataset_b_bundle.frame)

        # Execute the original GPU pipeline stages.
        rl_cli.global_config.USE_GPU_COMPARISON = pipeline_cfg.use_gpu

        start_time = time.time()
        recA_gdf = rl_cli.loadDataset.load_data_set(
            pipeline_cfg.dataset_a,
            pipeline_cfg.id_column,
            pipeline_cfg.attributes,
        )
        recB_gdf = rl_cli.loadDataset.load_data_set(
            pipeline_cfg.dataset_b,
            pipeline_cfg.id_column,
            pipeline_cfg.attributes,
        )
        true_match_set = rl_cli.loadDataset.load_truth_data(pipeline_cfg.truth)
        loading_time = time.time() - start_time

        candidate_pairs_gdf = rl_cli._generate_candidate_pairs(pipeline_cfg, recA_gdf, recB_gdf)
        raw_candidate_pairs = len(candidate_pairs_gdf)
        if candidate_pairs_gdf.empty:
            raise RuntimeError("Record linkage produced no candidate pairs; review blocking configuration.")

        start_time = time.time()
        comparison_pairs = rl_cli._resolve_comparison_pairs(pipeline_cfg)
        sim_vectors_gdf = rl_cli.comparison.compare_pairs(
            candidate_pairs_gdf,
            recA_gdf,
            recB_gdf,
            comparison_pairs,
        )
        comparison_time = time.time() - start_time
        if sim_vectors_gdf.empty:
            raise RuntimeError("Similarity computation returned no vectors; check comparison settings.")

        filtered_vectors_gdf = rl_cli._apply_filter_profile(sim_vectors_gdf, pipeline_cfg.filters)
        filtered_candidate_pairs = len(filtered_vectors_gdf)
        if filtered_vectors_gdf.empty:
            raise RuntimeError("All candidate pairs were removed by precision filters.")

        start_time = time.time()
        class_match_set, class_nonmatch_set = rl_cli.classification.supervisedMLClassify(
            filtered_vectors_gdf,
            true_match_set,
            n_estimators=pipeline_cfg.classification.n_estimators,
            threshold=pipeline_cfg.classification.base_threshold,
            threshold_offset=pipeline_cfg.classification.threshold_offset,
            min_precision=pipeline_cfg.classification.min_precision,
            min_recall=pipeline_cfg.classification.min_recall,
            precision_beta=pipeline_cfg.classification.precision_beta,
        )
        classification_time = time.time() - start_time

        num_comparisons = filtered_candidate_pairs
        all_comparisons = int(len(recA_gdf) * len(recB_gdf))

        conf_matrix = rl_cli.evaluation.confusion_matrix(
            class_match_set,
            class_nonmatch_set,
            true_match_set,
            all_comparisons,
        )
        linkage_metrics = {
            "accuracy": float(rl_cli.evaluation.accuracy(conf_matrix)),
            "precision": float(rl_cli.evaluation.precision(conf_matrix)),
            "recall": float(rl_cli.evaluation.recall(conf_matrix)),
            "f_measure": float(rl_cli.evaluation.fmeasure(conf_matrix)),
        }

        blocking_metrics = {
            "initial_candidates": int(raw_candidate_pairs),
            "filtered_candidates": int(filtered_candidate_pairs),
            "reduction_ratio": float(rl_cli.evaluation.reduction_ratio(num_comparisons, all_comparisons)),
            "pairs_completeness": float(rl_cli.evaluation.pairs_completeness(filtered_vectors_gdf, true_match_set)),
            "pairs_quality": float(rl_cli.evaluation.pairs_quality(filtered_vectors_gdf, true_match_set)),
        }

        total_runtime = loading_time + comparison_time + classification_time
        runtime = {
            "loading_seconds": float(loading_time),
            "comparison_seconds": float(comparison_time),
            "classification_seconds": float(classification_time),
            "total_seconds": float(total_runtime),
        }

        output_path_final = Path(pipeline_cfg.output_csv).expanduser().resolve()
        rl_cli.saveLinkResult.save_linkage_set(str(output_path_final), class_match_set)

        return RecordLinkageResult(
            dataset_key=pipeline_cfg.dataset_key,
            dataset_a_path=Path(pipeline_cfg.dataset_a).expanduser().resolve(),
            dataset_b_path=Path(pipeline_cfg.dataset_b).expanduser().resolve(),
            truth_path=Path(pipeline_cfg.truth).expanduser().resolve() if pipeline_cfg.truth else None,
            output_path=output_path_final,
            id_column=pipeline_cfg.id_column,
            attributes=list(pipeline_cfg.attributes),
            match_count=len(class_match_set),
            non_match_count=len(class_nonmatch_set),
            true_match_count=len(true_match_set),
            candidate_pairs=raw_candidate_pairs,
            filtered_pairs=filtered_candidate_pairs,
            blocking_metrics=blocking_metrics,
            linkage_metrics=linkage_metrics,
            runtime=runtime,
            analysis_a=analysis_a,
            analysis_b=analysis_b,
        )

    @staticmethod
    def _apply_overrides(
        pipeline_cfg: PipelineConfig,
        *,
        output_path: Optional[Path],
        use_gpu: Optional[bool],
        skip_filters: bool,
    ) -> PipelineConfig:
        """Mimic the standalone CLI overrides inside Prism."""
        from dataclasses import replace

        cfg = pipeline_cfg

        if output_path is not None:
            cfg = replace(cfg, output_csv=str(output_path.expanduser()))

        if use_gpu is not None and use_gpu != cfg.use_gpu:
            cfg = replace(cfg, use_gpu=use_gpu)

        if skip_filters and cfg.filters.enabled:
            cfg = replace(cfg, filters=replace(cfg.filters, enabled=False))

        return cfg
