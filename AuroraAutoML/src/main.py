"""CLI entrypoint for the GPU AutoML pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .pipeline import AutoMLPipeline

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-accelerated AutoML pipeline.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    console.print(f"Dataset shape: {result.analysis_report.dataset_shape[0]} rows × {result.analysis_report.dataset_shape[1]} columns")
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


if __name__ == "__main__":
    main()
