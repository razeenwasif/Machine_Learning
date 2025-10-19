"""Configurable entry point for the record linkage pipeline.

This script orchestrates the full workflow:

1. Load datasets and the optional ground-truth match file.
2. Block/partition the data, then generate candidate pairs via ANN.
3. Compute similarity vectors for each candidate pair.
4. Apply optional precision-focused filters to reduce false positives.
5. Train and apply a supervised classifier to score candidate pairs.
6. Evaluate the results (when a truth file is available) and write matches to disk.

Pipeline settings are defined in ``config/pipeline.toml``. Users can add or tweak
dataset presets, adjust blocking/ANN parameters, configure comparison functions,
and tune classification thresholds without modifying code.
"""

# conda activate rapids-rl 

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import replace
from typing import List, Optional, Tuple

import cupy
import cudf

from . import blocking
from . import classification
from . import comparison
from . import config as global_config
from . import evaluation
from . import loadDataset
from . import saveLinkResult
from .pipeline_config import (
    ConditionConfig,
    FilterGroupConfig,
    FilterProfileConfig,
    PipelineConfig,
    PipelineConfigError,
    load_pipeline_config,
    list_available_datasets,
)

# conda run -n rapids-rl python src/recordLinkage.py

# --- Setup Logging ---
logging.basicConfig(level=global_config.LOG_LEVEL, format=global_config.LOG_FORMAT)
LOGGER = logging.getLogger("recordLinkage")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated record linkage pipeline powered by RAPIDS."
    )
    parser.add_argument(
        "--config",
        default="../config/pipeline.toml",
        help="Path to the pipeline configuration file (default: ../config/pipeline.toml).",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset preset key defined inside the configuration file.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print the dataset presets found in the configuration file and exit.",
    )
    parser.add_argument(
        "--output",
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
        help="Disable high-precision filters (useful when tuning thresholds).",
    )
    parser.add_argument(
        "legacy_dataset",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_gpu=None)
    return parser.parse_args()


def _resolve_dataset_argument(args: argparse.Namespace) -> Optional[str]:
    """Support both --dataset and the legacy positional dataset argument."""
    if args.dataset and args.legacy_dataset:
        LOGGER.warning(
            "Both --dataset=%s and positional dataset '%s' supplied. "
            "The --dataset flag takes precedence.",
            args.dataset,
            args.legacy_dataset,
        )
        return args.dataset
    if args.dataset:
        return args.dataset
    if args.legacy_dataset:
        LOGGER.info(
            "Using positional dataset argument '%s'. "
            "Prefer passing --dataset=<name> for clarity.",
            args.legacy_dataset,
        )
        return args.legacy_dataset
    return None


def _resolve_comparison_pairs(pipeline_cfg: PipelineConfig) -> List[Tuple]:
    """Convert comparison function names declared in the config into callables."""
    resolved_pairs: List[Tuple] = []
    missing_functions: List[str] = []

    for pair_cfg in pipeline_cfg.comparisons:
        func = getattr(comparison, pair_cfg.function, None)
        if func is None:
            missing_functions.append(pair_cfg.function)
            continue
        resolved_pairs.append((func, pair_cfg.attr_a, pair_cfg.attr_b))

    if missing_functions:
        raise PipelineConfigError(
            "Comparison functions not found in comparison.py: "
            + ", ".join(sorted(set(missing_functions)))
        )
    return resolved_pairs


def _full_true_series(length: int) -> cudf.Series:
    if length == 0:
        return cudf.Series([], dtype="bool")
    return cudf.Series(cupy.ones(length, dtype=cupy.bool_))


def _evaluate_condition(
    sim_vectors: cudf.DataFrame,
    condition: ConditionConfig,
    missing_columns_cache: set,
) -> cudf.Series:
    """Return a boolean mask indicating which rows satisfy the given condition."""
    column = condition.column
    if column not in sim_vectors.columns:
        if column not in missing_columns_cache:
            LOGGER.warning(
                "Precision filter skipped for missing column '%s'. "
                "Consider removing or correcting the condition in the config.",
                column,
            )
            missing_columns_cache.add(column)
        return _full_true_series(len(sim_vectors))

    col = sim_vectors[column].fillna(0.0)
    op = condition.operator
    value = condition.value

    if op == ">=":
        return col >= value
    if op == ">":
        return col > value
    if op == "<=":
        return col <= value
    if op == "<":
        return col < value
    if op == "==":
        return col == value
    if op == "!=":
        return col != value
    raise PipelineConfigError(f"Unsupported operator '{op}' in condition for column {column}")


def _combine_group_masks(
    sim_vectors: cudf.DataFrame,
    group_cfg: FilterGroupConfig,
    missing_columns_cache: set,
) -> cudf.Series:
    """Evaluate a filter group returning a mask of rows passing the group."""
    if sim_vectors.empty:
        return _full_true_series(0)

    all_mask: Optional[cudf.Series] = None
    for condition in group_cfg.all_conditions:
        condition_mask = _evaluate_condition(sim_vectors, condition, missing_columns_cache)
        all_mask = condition_mask if all_mask is None else all_mask & condition_mask

    if all_mask is None:
        all_mask = _full_true_series(len(sim_vectors))

    if not group_cfg.any_conditions:
        return all_mask if group_cfg.min_any == 0 else cudf.Series(
            cupy.zeros(len(sim_vectors), dtype=cupy.bool_)
        )

    condition_masks = [
        _evaluate_condition(sim_vectors, condition, missing_columns_cache).astype("int8")
        for condition in group_cfg.any_conditions
    ]
    counts_df = cudf.concat(condition_masks, axis=1)
    counts = counts_df.sum(axis=1)
    any_mask = counts >= group_cfg.min_any
    return all_mask & any_mask


def _apply_filter_profile(
    sim_vectors_gdf: cudf.DataFrame,
    filter_cfg: FilterProfileConfig,
) -> cudf.DataFrame:
    """Apply precision filters defined in the configuration."""
    if sim_vectors_gdf.empty:
        LOGGER.info("Skipping precision filters (%s); no candidate pairs available.", filter_cfg.name)
        return sim_vectors_gdf

    if not filter_cfg.enabled:
        LOGGER.info("Precision filters disabled for profile '%s'.", filter_cfg.name)
        return sim_vectors_gdf

    missing_columns_cache: set = set()

    enforce_mask: Optional[cudf.Series] = None
    for condition in filter_cfg.enforce_all:
        cond_mask = _evaluate_condition(sim_vectors_gdf, condition, missing_columns_cache)
        enforce_mask = cond_mask if enforce_mask is None else enforce_mask & cond_mask
    if enforce_mask is None:
        enforce_mask = _full_true_series(len(sim_vectors_gdf))

    if not filter_cfg.groups:
        final_mask = enforce_mask
    else:
        group_masks = [
            _combine_group_masks(sim_vectors_gdf, group_cfg, missing_columns_cache)
            for group_cfg in filter_cfg.groups
        ]
        combined_groups = group_masks[0]
        for group_mask in group_masks[1:]:
            combined_groups = combined_groups | group_mask
        final_mask = enforce_mask & combined_groups

    filtered = sim_vectors_gdf[final_mask]
    LOGGER.info(
        "Precision filters (%s) kept %d of %d candidate pairs (%.2f%%).",
        filter_cfg.name,
        len(filtered),
        len(sim_vectors_gdf),
        100.0 * len(filtered) / max(1, len(sim_vectors_gdf)),
    )
    return filtered.reset_index(drop=True)


def _cartesian_candidate_pairs(
    gdf_a: cudf.DataFrame,
    gdf_b: cudf.DataFrame,
) -> cudf.DataFrame:
    """Compute the Cartesian product of two partitions when ANN is disabled."""
    if gdf_a.empty or gdf_b.empty:
        return cudf.DataFrame({"rec_id_A": [], "rec_id_B": []})

    tmp_a = gdf_a.reset_index().rename(columns={"index": "rec_id_A"})
    tmp_b = gdf_b.reset_index().rename(columns={"index": "rec_id_B"})
    tmp_a["__tmp__"] = 1
    tmp_b["__tmp__"] = 1

    merged = tmp_a.merge(tmp_b, on="__tmp__", how="inner")
    result = merged[["rec_id_A", "rec_id_B"]]

    del tmp_a, tmp_b, merged
    return result


def _generate_candidate_pairs(
    pipeline_cfg: PipelineConfig,
    recA_gdf: cudf.DataFrame,
    recB_gdf: cudf.DataFrame,
) -> cudf.DataFrame:
    """Run partitioning and ANN candidate generation based on the configuration."""
    blocking_cfg = pipeline_cfg.blocking
    partition_attrs = blocking_cfg.partition_attributes

    start_time = time.time()
    if partition_attrs:
        LOGGER.info("Partitioning datasets by %s", partition_attrs)
        blocks_A = blocking.simpleBlocking(recA_gdf, partition_attrs)
        blocks_B = blocking.simpleBlocking(recB_gdf, partition_attrs)
        common_keys = sorted(set(blocks_A.keys()) & set(blocks_B.keys()))
    else:
        LOGGER.info("No partition attributes configured; using a single global block.")
        blocks_A = {"__all__": recA_gdf.index.to_arrow().to_pylist()}
        blocks_B = {"__all__": recB_gdf.index.to_arrow().to_pylist()}
        common_keys = ["__all__"]

    if not common_keys:
        LOGGER.warning("No overlapping partitions were found; no candidate pairs generated.")
        return cudf.DataFrame({"rec_id_A": [], "rec_id_B": []})

    candidate_pairs_list: List[cudf.DataFrame] = []
    ann_cfg = blocking_cfg.ann

    for idx, key in enumerate(common_keys, start=1):
        LOGGER.info("  Processing partition %d/%d: %s", idx, len(common_keys), key)
        rec_ids_A = blocks_A[key]
        rec_ids_B = blocks_B[key]

        temp_gdf_A = recA_gdf.loc[rec_ids_A]
        temp_gdf_B = recB_gdf.loc[rec_ids_B]

        if ann_cfg.enabled:
            candidate_pairs_gdf = blocking.ann_candidate_generation(
                temp_gdf_A,
                temp_gdf_B,
                k=ann_cfg.k_neighbors,
                blk_attr_list=ann_cfg.attributes,
                sim_threshold=ann_cfg.similarity_threshold,
            )
        else:
            candidate_pairs_gdf = _cartesian_candidate_pairs(temp_gdf_A, temp_gdf_B)

        candidate_pairs_list.append(candidate_pairs_gdf)

    if not candidate_pairs_list:
        return cudf.DataFrame({"rec_id_A": [], "rec_id_B": []})

    candidate_pairs = cudf.concat(candidate_pairs_list, ignore_index=True)
    candidate_pairs = candidate_pairs.drop_duplicates()

    blocking_duration = time.time() - start_time
    LOGGER.info(
        "Generated %d candidate pairs across %d partitions in %.3f seconds.",
        len(candidate_pairs),
        len(common_keys),
        blocking_duration,
    )
    return candidate_pairs


def _apply_overrides(
    pipeline_cfg: PipelineConfig,
    args: argparse.Namespace,
) -> PipelineConfig:
    """Apply CLI overrides to the loaded pipeline configuration."""
    cfg = pipeline_cfg

    if args.output:
        output_path = os.path.normpath(os.path.expanduser(args.output))
        cfg = replace(cfg, output_csv=output_path)

    if args.use_gpu is not None and args.use_gpu != cfg.use_gpu:
        cfg = replace(cfg, use_gpu=args.use_gpu)

    if args.skip_filters and cfg.filters.enabled:
        filters_disabled = replace(cfg.filters, enabled=False)
        cfg = replace(cfg, filters=filters_disabled)

    return cfg


def main() -> int:
    args = _parse_args()

    if args.list_datasets:
        dataset_keys = list_available_datasets(args.config)
        if not dataset_keys:
            print("No datasets found in", args.config)
        else:
            print("Datasets defined in", args.config)
            for key in dataset_keys:
                print("  -", key)
        return 0

    dataset_key = _resolve_dataset_argument(args)

    try:
        pipeline_cfg = load_pipeline_config(args.config, dataset_key=dataset_key)
    except PipelineConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 1

    pipeline_cfg = _apply_overrides(pipeline_cfg, args)

    LOGGER.info(
        "Running dataset preset '%s' using configuration '%s'.",
        pipeline_cfg.dataset_key,
        os.path.abspath(args.config),
    )

    LOGGER.info("GPU comparisons %s.", "enabled" if pipeline_cfg.use_gpu else "disabled")
    global_config.USE_GPU_COMPARISON = pipeline_cfg.use_gpu

    comparison_pairs = _resolve_comparison_pairs(pipeline_cfg)

    # Step 1: Load datasets
    start_time = time.time()
    recA_gdf = loadDataset.load_data_set(
        pipeline_cfg.dataset_a,
        pipeline_cfg.id_column,
        pipeline_cfg.attributes,
    )
    recB_gdf = loadDataset.load_data_set(
        pipeline_cfg.dataset_b,
        pipeline_cfg.id_column,
        pipeline_cfg.attributes,
    )
    true_match_set = loadDataset.load_truth_data(pipeline_cfg.truth)

    loading_time = time.time() - start_time
    LOGGER.info("Data loading finished in %.3f seconds.", loading_time)

    # Step 2: Blocking + candidate generation
    candidate_pairs_gdf = _generate_candidate_pairs(pipeline_cfg, recA_gdf, recB_gdf)
    if candidate_pairs_gdf.empty:
        LOGGER.info("No candidate pairs generated; exiting.")
        return 0

    # Step 3: Compare candidate pairs
    start_time = time.time()
    sim_vectors_gdf = comparison.compare_pairs(
        candidate_pairs_gdf, recA_gdf, recB_gdf, comparison_pairs
    )
    comparison_time = time.time() - start_time
    LOGGER.info("Computed similarity vectors in %.3f seconds.", comparison_time)

    if sim_vectors_gdf.empty:
        LOGGER.info("No similarity vectors produced; exiting.")
        return 0

    # Step 4: Precision-focused filtering
    sim_vectors_gdf = _apply_filter_profile(sim_vectors_gdf, pipeline_cfg.filters)
    if sim_vectors_gdf.empty:
        LOGGER.info("All candidate pairs filtered out; exiting.")
        return 0

    # Step 5: Classification
    start_time = time.time()
    class_match_set, class_nonmatch_set = classification.supervisedMLClassify(
        sim_vectors_gdf,
        true_match_set,
        n_estimators=pipeline_cfg.classification.n_estimators,
        threshold=pipeline_cfg.classification.base_threshold,
        threshold_offset=pipeline_cfg.classification.threshold_offset,
        min_precision=pipeline_cfg.classification.min_precision,
        min_recall=pipeline_cfg.classification.min_recall,
        precision_beta=pipeline_cfg.classification.precision_beta,
    )
    classification_time = time.time() - start_time
    LOGGER.info("Classification stage completed in %.3f seconds.", classification_time)

    # Step 6: Evaluation (requires truth data)
    num_comparisons = len(sim_vectors_gdf)
    all_comparisons = len(recA_gdf) * len(recB_gdf)
    evaluation.evaluate_blocking(sim_vectors_gdf, true_match_set, num_comparisons, all_comparisons)
    evaluation.evaluate_linkage(class_match_set, class_nonmatch_set, true_match_set, all_comparisons)

    total_runtime = loading_time + comparison_time + classification_time
    LOGGER.info("Total end-to-end runtime: %.3f seconds.", total_runtime)

    # Step 7: Persist results
    saveLinkResult.save_linkage_set(pipeline_cfg.output_csv, class_match_set)
    LOGGER.info("Saved %d matched pairs to '%s'.", len(class_match_set), pipeline_cfg.output_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
