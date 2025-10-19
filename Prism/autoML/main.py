"""Unified CLI entry point for Prism AutoML and record linkage pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console
from rich.table import Table

from .pipeline import AutoMLPipeline

console = Console()


def _add_automl_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=str, required=True, help="Path to input dataset (CSV, JSON, Parquet).")
    parser.add_argument("--target", type=str, default=None, help="Target column for supervised tasks.")
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "regression", "classification", "clustering"],
        help="Task override. Defaults to auto-detection.",
    )
    parser.add_argument("--max-trials", type=int, default=20, help="Maximum Optuna trials.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA behaviour (may reduce performance).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU execution even if a GPU is available.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to dump pipeline results as JSON.",
    )


def _add_link_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the record linkage pipeline TOML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset preset key defined in the configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output CSV destination defined in the configuration.",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="Force GPU comparisons even if the config disables them.",
    )
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disable GPU comparisons regardless of the config defaults.",
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Disable precision-focused filters (useful while tuning thresholds).",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List dataset presets defined in the configuration file and exit.",
    )
    parser.set_defaults(use_gpu=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prism pipelines (AutoML and record linkage).")
    subparsers = parser.add_subparsers(dest="command")

    automl_parser = subparsers.add_parser("automl", help="Run the AutoML workflow.")
    _add_automl_arguments(automl_parser)

    link_parser = subparsers.add_parser("link", help="Run the GPU record linkage workflow.")
    _add_link_arguments(link_parser)

    return parser


def _run_automl(args: argparse.Namespace) -> None:
    pipeline = AutoMLPipeline(
        seed=args.seed,
        deterministic=args.deterministic,
        max_trials=args.max_trials,
        prefer_gpu=not args.no_gpu,
    )

    result = pipeline.run(
        data_path=args.data,
        target=args.target,
        task=args.task,
        test_size=args.test_size,
        max_trials=args.max_trials,
    )

    rows, cols = result.analysis_report.dataset_shape
    console.print(f"Dataset shape: {rows} rows × {cols} columns")
    if result.cleaning_report.has_changes():
        console.print("[bold yellow]Cleaning steps applied:[/bold yellow]")
        for step in result.cleaning_report.applied_steps:
            console.print(f"  - {step}")

    console.print(f"[bold green]Best model:[/bold green] {result.model_name}")
    console.print(f"Task: {result.task}")
    metrics_table = Table(title="Evaluation Metrics")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value")
    for metric, value in result.metrics.items():
        metrics_table.add_row(metric, f"{value:.4f}")
    console.print(metrics_table)

    console.print("Best configuration:")
    for key, value in result.best_config.items():
        console.print(f"  - {key}: {value}")

    if result.analysis_report.correlation_pairs:
        console.print("\nTop feature correlations:")
        corr_table = Table(show_header=True, header_style="bold magenta")
        corr_table.add_column("Feature A")
        corr_table.add_column("Feature B")
        corr_table.add_column("Correlation", justify="right")
        for correlation in result.analysis_report.correlation_pairs[:5]:
            corr_table.add_row(
                correlation.feature_a,
                correlation.feature_b,
                f"{correlation.correlation:.3f}",
            )
        console.print(corr_table)

    if result.analysis_report.target_summary:
        summary = result.analysis_report.target_summary.to_dict()
        console.print("\nTarget summary:")
        for key, value in summary["distribution"].items():
            console.print(f"  - {key}: {value:.3f}")

    if args.output_json:
        args.output_json.write_text(pipeline.to_json(result))
        console.print(f"Saved results to {args.output_json}")


def _render_notes(label: str, notes: Iterable[str]) -> None:
    notes = list(notes)
    if not notes:
        return
    console.print(f"{label} notes:")
    for note in notes[:3]:
        console.print(f"  - {note}")


def _run_link(args: argparse.Namespace) -> None:
    from recordLinkage.src import PipelineConfigError, list_available_datasets

    from .linkage import RecordLinkageDependencyError, RecordLinkagePipeline

    config_path: Optional[Path] = args.config
    if args.list_datasets:
        pipeline = RecordLinkagePipeline(config_path=config_path)
        dataset_keys = list_available_datasets(str(pipeline.config_path))
        if not dataset_keys:
            console.print(f"No datasets defined in {pipeline.config_path}")
        else:
            console.print(f"Datasets defined in {pipeline.config_path}:")
            for key in dataset_keys:
                console.print(f"  - {key}")
        return

    try:
        pipeline = RecordLinkagePipeline(config_path=config_path)
        result = pipeline.run(
            dataset_key=args.dataset,
            output_path=args.output,
            use_gpu=args.use_gpu,
            skip_filters=args.skip_filters,
        )
    except RecordLinkageDependencyError as exc:
        console.print(f"[bold red]Record linkage unavailable:[/bold red] {exc}")
        return
    except PipelineConfigError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return

    rows_a, cols_a = result.analysis_a.dataset_shape
    rows_b, cols_b = result.analysis_b.dataset_shape
    console.print(f"[bold green]Record linkage completed for preset:[/bold green] {result.dataset_key}")
    console.print(f"Dataset A ({rows_a}×{cols_a}): {result.dataset_a_path}")
    _render_notes("Dataset A", result.analysis_a.notes)
    console.print(f"Dataset B ({rows_b}×{cols_b}): {result.dataset_b_path}")
    _render_notes("Dataset B", result.analysis_b.notes)
    if result.truth_path:
        console.print(f"Truth matches: {result.true_match_count} rows ({result.truth_path})")

    console.print(
        f"Matched pairs: {result.match_count} (non-matches: {result.non_match_count}) "
        f"→ saved to {result.output_path}"
    )

    blocking_table = Table(title="Blocking Metrics")
    blocking_table.add_column("Metric")
    blocking_table.add_column("Value", justify="right")
    blocking_table.add_row("Initial candidates", f"{int(result.blocking_metrics['initial_candidates']):,}")
    blocking_table.add_row("Filtered candidates", f"{int(result.blocking_metrics['filtered_candidates']):,}")
    blocking_table.add_row("Reduction ratio", f"{result.blocking_metrics['reduction_ratio']:.4f}")
    blocking_table.add_row("Pairs completeness", f"{result.blocking_metrics['pairs_completeness']:.4f}")
    blocking_table.add_row("Pairs quality", f"{result.blocking_metrics['pairs_quality']:.4f}")
    console.print(blocking_table)

    linkage_table = Table(title="Linkage Metrics")
    linkage_table.add_column("Metric")
    linkage_table.add_column("Value", justify="right")
    for metric, value in result.linkage_metrics.items():
        linkage_table.add_row(metric.replace("_", " ").title(), f"{value:.4f}")
    console.print(linkage_table)

    runtime_table = Table(title="Stage Runtime (seconds)")
    runtime_table.add_column("Stage")
    runtime_table.add_column("Seconds", justify="right")
    runtime_table.add_row("Loading", f"{result.runtime['loading_seconds']:.3f}")
    runtime_table.add_row("Similarity", f"{result.runtime['comparison_seconds']:.3f}")
    runtime_table.add_row("Classification", f"{result.runtime['classification_seconds']:.3f}")
    runtime_table.add_row("Total", f"{result.runtime['total_seconds']:.3f}")
    console.print(runtime_table)


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    if not raw_argv:
        raw_argv = ["automl"]
    else:
        first = raw_argv[0]
        if first not in {"automl", "link"}:
            if first in ("-h", "--help"):
                parser.parse_args(raw_argv)
                return
            if first.startswith("-"):
                raw_argv = ["automl", *raw_argv]
            else:
                raw_argv = ["automl", *raw_argv]

    args = parser.parse_args(raw_argv)
    command = args.command or "automl"

    if command == "automl":
        _run_automl(args)
    elif command == "link":
        _run_link(args)
    else:  # pragma: no cover - defensive fallback
        parser.error(f"Unknown command '{command}'")


if __name__ == "__main__":
    main()
