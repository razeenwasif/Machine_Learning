"""High-level orchestration for the GPU AutoML pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from rich.console import Console
from sklearn.model_selection import train_test_split
import pandas as pd

from .analysis import AnalysisReport, CleaningReport, DataAnalyzer, DataCleaner
from .data.loaders import DatasetBundle, load_dataset
from .hpo import HPOResult, HyperparameterOptimizer
from .models.model_factory import candidates_for_task, instantiate_model
from .utils.device import DeviceConfig, configure_runtime, release_gpu_cache
from .utils.metrics import (
    classification_metrics,
    clustering_metrics,
    pick_primary_metric,
    regression_metrics,
)
from .utils.preprocessing import Preprocessor

console = Console()


@dataclass
class PipelineResult:
    task: str
    model_name: str
    metrics: Dict[str, float]
    best_config: Dict
    hpo_score: float
    analysis_report: AnalysisReport
    cleaning_report: CleaningReport


class AutoMLPipeline:
    """End-to-end pipeline for dataset-driven model selection."""

    def __init__(
        self,
        seed: Optional[int] = None,
        deterministic: bool = False,
        max_trials: int = 20,
        prefer_gpu: bool = True,
    ) -> None:
        self.seed = seed
        self.deterministic = deterministic
        self.max_trials = max_trials
        self.prefer_gpu = prefer_gpu
        self.device_config: DeviceConfig = configure_runtime(
            seed=seed,
            deterministic=deterministic,
            prefer_gpu=prefer_gpu,
        )

    def run(
        self,
        data_path: str,
        target: Optional[str] = None,
        task: str = "auto",
        test_size: float = 0.2,
        max_trials: Optional[int] = None,
    ) -> PipelineResult:
        dataset = load_dataset(data_path, target)

        analyzer = DataAnalyzer()
        analysis_report = analyzer.analyze(dataset.frame, target)

        cleaner = DataCleaner()
        cleaned_frame, cleaning_report = cleaner.clean(dataset.frame, analysis_report)
        if cleaning_report.has_changes():
            analysis_report.notes.append("Automatic cleaning applied before training.")
            console.log("Applied automatic data cleaning steps.")

        dataset = DatasetBundle(frame=cleaned_frame, path=dataset.path, target=dataset.target)
        resolved_task = self._infer_task(dataset, task)
        console.log(f"Detected task: [bold]{resolved_task}[/bold]")

        if resolved_task != "clustering" and not target:
            raise ValueError("Target column is required for supervised tasks.")

        if max_trials is None:
            max_trials = self.max_trials

        preprocessor = Preprocessor(target_column=target, task_type=resolved_task)

        train_df, test_df = self._train_test_split(dataset, resolved_task, test_size)
        console.log(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

        train_features_tensor, train_targets_tensor = preprocessor.fit_transform(train_df)
        test_features_tensor, test_targets_tensor = preprocessor.transform(test_df)

        (
            hpo_train_features,
            hpo_val_features,
            hpo_train_targets,
            hpo_val_targets,
        ) = self._create_validation_split(
            train_features_tensor,
            train_targets_tensor,
            resolved_task,
        )

        metric_fn, direction = self._select_primary_metric(resolved_task, hpo_val_features, hpo_val_targets)

        candidates = [
            (model_cls.name, (model_cls, config))
            for model_cls, config in candidates_for_task(resolved_task)
        ]

        optimizer = HyperparameterOptimizer(
            task=resolved_task,
            device=self.device_config.device,
            metric_fn=metric_fn,
            direction=direction,
            n_trials=max_trials,
            seed=self.seed,
            context={
                "n_samples": int(hpo_train_features.shape[0]),
            },
        )

        hpo_result = optimizer.optimize(
            candidates=candidates,
            train_data=(hpo_train_features, hpo_train_targets),
            val_data=(hpo_val_features, hpo_val_targets),
        )
        console.log(f"HPO best model: [bold]{hpo_result.model_name}[/bold] (score={hpo_result.best_score:.4f})")

        final_model = self._train_best_model(
            result=hpo_result,
            task=resolved_task,
            train_features=train_features_tensor,
            train_targets=train_targets_tensor,
        )

        metrics = self._evaluate(final_model, resolved_task, test_features_tensor, test_targets_tensor)

        final_model.cleanup()
        release_gpu_cache()

        return PipelineResult(
            task=resolved_task,
            model_name=hpo_result.model_name,
            metrics=metrics,
            best_config=hpo_result.config,
            hpo_score=hpo_result.best_score,
            analysis_report=analysis_report,
            cleaning_report=cleaning_report,
        )

    def _train_test_split(
        self,
        dataset: DatasetBundle,
        task: str,
        test_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        frame = dataset.frame
        if task == "clustering":
            train_df, test_df = train_test_split(frame, test_size=test_size, random_state=self.seed)
        else:
            target_col = dataset.target
            stratify = None
            if task == "classification" and target_col:
                stratify = frame[target_col]
            train_df, test_df = train_test_split(
                frame,
                test_size=test_size,
                random_state=self.seed,
                stratify=stratify,
            )
        return train_df, test_df

    def _create_validation_split(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor],
        task: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        features_np = features.numpy()

        if task == "clustering":
            train_np, val_np = train_test_split(features_np, test_size=0.2, random_state=self.seed)
            return (
                torch.from_numpy(train_np).float(),
                torch.from_numpy(val_np).float(),
                None,
                None,
            )

        if targets is None:
            raise ValueError("Supervised tasks require targets.")

        targets_np = targets.numpy()
        stratify = None
        if task == "classification":
            stratify = targets_np
        train_X, val_X, train_y, val_y = train_test_split(
            features_np,
            targets_np,
            test_size=0.2,
            random_state=self.seed,
            stratify=stratify,
        )

        train_features_tensor = torch.from_numpy(train_X).float()
        val_features_tensor = torch.from_numpy(val_X).float()

        if targets.dtype in (torch.long, torch.int64):
            train_targets_tensor = torch.from_numpy(train_y).long()
            val_targets_tensor = torch.from_numpy(val_y).long()
        else:
            train_targets_tensor = torch.from_numpy(train_y).float()
            val_targets_tensor = torch.from_numpy(val_y).float()

        return train_features_tensor, val_features_tensor, train_targets_tensor, val_targets_tensor

    def _select_primary_metric(
        self,
        task: str,
        val_features: torch.Tensor,
        val_targets: Optional[torch.Tensor],
    ):
        if task == "regression":
            metric_name, metric_fn = pick_primary_metric(task)
            console.log(f"Primary metric: [bold]{metric_name}[/bold] (minimise)")
            return metric_fn, "minimize"
        if task == "classification":
            metric_name, metric_fn = pick_primary_metric(task)
            console.log(f"Primary metric: [bold]{metric_name}[/bold] (maximise)")
            return metric_fn, "maximize"
        if task == "clustering":
            metrics_map = clustering_metrics(val_features.numpy())
            metric_fn = metrics_map["silhouette"]
            console.log("Primary metric: [bold]silhouette[/bold] (maximise)")
            return metric_fn, "maximize"
        raise ValueError(f"Unknown task: {task}")

    def _train_best_model(
        self,
        result: HPOResult,
        task: str,
        train_features: torch.Tensor,
        train_targets: Optional[torch.Tensor],
    ):
        model_cls = None
        for candidate_cls, _ in candidates_for_task(task):
            if candidate_cls.name == result.model_name:
                model_cls = candidate_cls
                break
        if model_cls is None:
            raise RuntimeError(f"Unable to locate model class for {result.model_name}")

        final_model = instantiate_model(model_cls, self.device_config.device, result.config)
        final_model.fit(train_features, train_targets)
        return final_model

    def _evaluate(
        self,
        model,
        task: str,
        test_features: torch.Tensor,
        test_targets: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if task == "clustering":
            predictions = model.predict(test_features).numpy()
            metrics_map = clustering_metrics(test_features.numpy())
            metrics = {
                name: fn(test_features.numpy(), predictions)
                for name, fn in metrics_map.items()
            }
            return metrics

        if test_targets is None:
            raise ValueError("Supervised evaluation requires targets.")

        predictions = model.predict(test_features).numpy()
        y_true = test_targets.numpy()

        if task == "regression":
            for name, fn in regression_metrics().items():
                metrics[name] = fn(y_true, predictions)
        elif task == "classification":
            for name, fn in classification_metrics().items():
                metrics[name] = fn(y_true, predictions)
        else:
            raise ValueError(f"Unsupported task for evaluation: {task}")

        return metrics

    def _infer_task(self, dataset: DatasetBundle, supplied_task: str) -> str:
        if supplied_task != "auto":
            return supplied_task
        if dataset.target is None:
            return "clustering"
        series = dataset.frame[dataset.target]
        if series.dtype.kind in {"i", "b"}:
            unique_values = series.nunique()
            if unique_values <= max(20, int(0.05 * len(series))):
                return "classification"
        if series.dtype.kind in {"O", "b"}:
            return "classification"
        return "regression"

    def to_json(self, result: PipelineResult) -> str:
        return json.dumps(
            {
                "task": result.task,
                "model": result.model_name,
                "metrics": result.metrics,
                "best_config": result.best_config,
                "hpo_score": result.hpo_score,
                "analysis": result.analysis_report.to_dict(),
                "cleaning": result.cleaning_report.to_dict(),
            },
            indent=2,
        )

    def assign_clusters(
        self,
        data_path: str,
        model_name: str,
        config: Dict,
        target: Optional[str] = None,
    ) -> pd.DataFrame:
        """Train the supplied clustering config on the full dataset and return assignments."""
        dataset = load_dataset(data_path, target)

        analyzer = DataAnalyzer()
        analysis_report = analyzer.analyze(dataset.frame, target)

        cleaner = DataCleaner()
        cleaned_frame, _ = cleaner.clean(dataset.frame, analysis_report)

        preprocessor = Preprocessor(target_column=target, task_type="clustering")
        features_tensor, _ = preprocessor.fit_transform(cleaned_frame)

        model_cls = None
        for candidate_cls, _ in candidates_for_task("clustering"):
            if candidate_cls.name == model_name:
                model_cls = candidate_cls
                break
        if model_cls is None:
            raise RuntimeError(f"Unable to locate model class for {model_name}")

        model = instantiate_model(model_cls, self.device_config.device, config)
        try:
            model.fit(features_tensor, None)
            labels = model.predict(features_tensor).numpy()
        finally:
            model.cleanup()
            release_gpu_cache()

        assignments = cleaned_frame.copy()
        assignments["cluster"] = labels
        return assignments
