"""Utilities for loading and validating the record linkage pipeline settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import os

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


class PipelineConfigError(Exception):
    """Raised when the pipeline configuration file is invalid."""


@dataclass(frozen=True)
class ComparisonPairConfig:
    """Mapping of comparison function names to attribute pairs."""

    function: str
    attr_a: str
    attr_b: str


@dataclass(frozen=True)
class ConditionConfig:
    """Single threshold condition referenced by the high-precision filters."""

    column: str
    operator: str
    value: float


@dataclass(frozen=True)
class FilterGroupConfig:
    """Disjunctive rule group; a record passes if all conditions and the any-list hold."""

    all_conditions: List[ConditionConfig] = field(default_factory=list)
    any_conditions: List[ConditionConfig] = field(default_factory=list)
    min_any: int = 0


@dataclass(frozen=True)
class FilterProfileConfig:
    """Collection of rule groups applied to candidate pairs before classification."""

    name: str
    enabled: bool
    enforce_all: List[ConditionConfig]
    groups: List[FilterGroupConfig]


@dataclass(frozen=True)
class BlockingANNConfig:
    """Settings controlling ANN candidate generation."""

    enabled: bool
    attributes: List[str]
    k_neighbors: int
    similarity_threshold: float


@dataclass(frozen=True)
class BlockingConfig:
    """Blocking stage configuration."""

    partition_attributes: List[str]
    ann: BlockingANNConfig


@dataclass(frozen=True)
class ClassificationConfig:
    """Parameters controlling the supervised classifier."""

    n_estimators: int
    base_threshold: float
    threshold_offset: float
    min_precision: float
    min_recall: float
    precision_beta: float


@dataclass(frozen=True)
class PipelineConfig:
    """Top level pipeline configuration used by the CLI entry point."""

    dataset_key: str
    dataset_a: str
    dataset_b: str
    truth: str
    id_column: str
    attributes: List[str]
    blocking: BlockingConfig
    comparisons: List[ComparisonPairConfig]
    filters: FilterProfileConfig
    classification: ClassificationConfig
    output_csv: str
    use_gpu: bool


def _parse_condition(expr: str) -> ConditionConfig:
    """Parse a simple condition string like ``sim_last_name>=0.75``."""
    valid_ops = ("<=", ">=", "<", ">", "==", "!=")
    for op in valid_ops:
        if op in expr:
            column, raw_val = expr.split(op, 1)
            try:
                value = float(raw_val.strip())
            except ValueError as exc:  # pragma: no cover - defensive logging
                raise PipelineConfigError(f"Invalid numeric value in condition '{expr}'") from exc
            column_name = column.strip()
            if not column_name:
                raise PipelineConfigError(f"Missing column name in condition '{expr}'")
            return ConditionConfig(column=column_name, operator=op, value=value)
    raise PipelineConfigError(
        f"Condition '{expr}' does not include a recognised operator "
        "(expected one of <=, >=, <, >, ==, !=)"
    )


def _parse_condition_list(entries: Optional[List[str]]) -> List[ConditionConfig]:
    if not entries:
        return []
    return [_parse_condition(expr) for expr in entries]


def _parse_filter_profile(name: str, data: Dict) -> FilterProfileConfig:
    enabled = bool(data.get("enabled", True))
    enforce_all_raw = data.get("enforce_all", [])
    groups_raw = data.get("groups", [])

    enforce_all = _parse_condition_list(enforce_all_raw)

    groups: List[FilterGroupConfig] = []
    for idx, group in enumerate(groups_raw):
        if not isinstance(group, dict):
            raise PipelineConfigError(
                f"Filter profile '{name}' group #{idx+1} must be a table/object."
            )
        all_conditions = _parse_condition_list(group.get("all", []))
        any_conditions = _parse_condition_list(group.get("any", []))
        min_any = int(group.get("min_any", 0))
        if min_any < 0:
            raise PipelineConfigError(
                f"Filter profile '{name}' group #{idx+1} has negative min_any {min_any}"
            )
        if min_any > 0 and not any_conditions:
            raise PipelineConfigError(
                f"Filter profile '{name}' group #{idx+1} specifies min_any "
                "but no 'any' conditions."
            )
        groups.append(
            FilterGroupConfig(
                all_conditions=all_conditions,
                any_conditions=any_conditions,
                min_any=min_any,
            )
        )

    return FilterProfileConfig(
        name=name,
        enabled=enabled,
        enforce_all=enforce_all,
        groups=groups,
    )


def _load_toml(path: str) -> Dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise PipelineConfigError(f"Configuration file '{path}' was not found.") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PipelineConfigError(f"Invalid TOML in '{path}': {exc}") from exc


def load_pipeline_config(config_path: str, dataset_key: Optional[str] = None) -> PipelineConfig:
    """Load the pipeline configuration file and return the resolved settings."""

    data = _load_toml(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    datasets = data.get("datasets")
    if not datasets or not isinstance(datasets, list):
        raise PipelineConfigError("No datasets defined in configuration.")

    default_dataset_key = data.get("default_dataset")
    if dataset_key is None:
        dataset_key = default_dataset_key
    if dataset_key is None and datasets:
        dataset_key = datasets[0].get("key")
    if dataset_key is None:
        raise PipelineConfigError("Unable to resolve dataset key to use.")

    dataset_lookup = {entry.get("key"): entry for entry in datasets if entry.get("key")}
    if dataset_key not in dataset_lookup:
        available = ", ".join(sorted(dataset_lookup))
        raise PipelineConfigError(
            f"Unknown dataset preset '{dataset_key}'. Available presets: {available}"
        )
    dataset = dataset_lookup[dataset_key]

    defaults = data.get("defaults", {})
    id_column = dataset.get("id_column", defaults.get("id_column"))
    if not id_column:
        raise PipelineConfigError("No id_column specified in defaults or dataset entry.")

    attribute_list = dataset.get("attributes", defaults.get("attributes", []))
    if not attribute_list:
        raise PipelineConfigError(
            f"No attribute list defined for dataset '{dataset_key}'. "
            "Add 'attributes' to the dataset entry or defaults."
        )

    output_from_dataset = dataset.get("output_csv")
    default_output_csv = defaults.get("output_csv")
    output_csv = output_from_dataset or default_output_csv
    if not output_csv:
        raise PipelineConfigError(
            "Output CSV path is not defined. "
            "Specify 'output_csv' under [defaults] or the dataset entry."
        )

    runtime_defaults = data.get("runtime", {})
    use_gpu = bool(dataset.get("use_gpu", runtime_defaults.get("use_gpu", True)))

    blocking_defaults = data.get("blocking", {}).get("defaults", {})
    partition_attrs = dataset.get(
        "blocking_partition_attributes", blocking_defaults.get("partition_attributes", [])
    )
    if partition_attrs is None:
        partition_attrs = []

    ann_defaults = blocking_defaults.get("ann", {})
    ann_enabled = bool(dataset.get("blocking_ann_enabled", ann_defaults.get("enabled", True)))
    ann_attributes = dataset.get(
        "blocking_ann_attributes", ann_defaults.get("attributes", [])
    ) or []
    ann_k = int(dataset.get("blocking_k_neighbors", ann_defaults.get("k_neighbors", 25)))
    ann_sim_threshold = float(
        dataset.get(
            "blocking_similarity_threshold", ann_defaults.get("similarity_threshold", 0.5)
        )
    )

    blocking_config = BlockingConfig(
        partition_attributes=list(partition_attrs),
        ann=BlockingANNConfig(
            enabled=ann_enabled,
            attributes=list(ann_attributes),
            k_neighbors=ann_k,
            similarity_threshold=ann_sim_threshold,
        ),
    )

    comparison_data = data.get("comparison", {})
    comparison_profiles = comparison_data.get("profiles", {})
    if not comparison_profiles:
        raise PipelineConfigError("No comparison profiles defined in configuration.")

    comparison_profile_name = dataset.get(
        "comparison_profile", comparison_data.get("default_profile")
    )
    if not comparison_profile_name:
        raise PipelineConfigError(
            "No comparison profile selected. Define comparison.default_profile "
            "or set comparison_profile in the dataset entry."
        )
    if comparison_profile_name not in comparison_profiles:
        available = ", ".join(sorted(comparison_profiles))
        raise PipelineConfigError(
            f"Comparison profile '{comparison_profile_name}' not found. "
            f"Available profiles: {available}"
        )

    comparison_pairs_raw = comparison_profiles[comparison_profile_name]
    if not isinstance(comparison_pairs_raw, list):
        raise PipelineConfigError(
            f"Comparison profile '{comparison_profile_name}' must be defined as an array of tables."
        )
    comparison_pairs: List[ComparisonPairConfig] = []
    for entry in comparison_pairs_raw:
        function_name = entry.get("function")
        attr_a = entry.get("attr_a")
        attr_b = entry.get("attr_b")
        if not function_name or not attr_a or not attr_b:
            raise PipelineConfigError(
                f"Invalid comparison pair in profile '{comparison_profile_name}': {entry}"
            )
        comparison_pairs.append(
            ComparisonPairConfig(function=function_name, attr_a=attr_a, attr_b=attr_b)
        )
    if not comparison_pairs:
        raise PipelineConfigError(
            f"Comparison profile '{comparison_profile_name}' does not contain any attribute pairs."
        )

    classification_data = data.get("classification", {})
    classification_profiles = classification_data.get("profiles", {})
    if not classification_profiles:
        raise PipelineConfigError("No classification profiles defined in configuration.")

    default_class_profile = classification_data.get("default_profile")
    classification_profile_name = dataset.get("classification_profile", default_class_profile)
    if not classification_profile_name:
        raise PipelineConfigError(
            "No classification profile selected. "
            "Set classification.default_profile or classification_profile in the dataset entry."
        )
    if classification_profile_name not in classification_profiles:
        available = ", ".join(sorted(classification_profiles))
        raise PipelineConfigError(
            f"Classification profile '{classification_profile_name}' not found. "
            f"Available profiles: {available}"
        )

    class_profile = classification_profiles[classification_profile_name]
    n_estimators = int(
        dataset.get("n_estimators", classification_data.get("n_estimators", 250))
    )
    base_threshold = float(
        dataset.get("base_threshold", classification_data.get("base_threshold", 0.40))
    )
    threshold_offset = float(class_profile.get("threshold_offset", 0.0))
    min_precision = float(class_profile.get("min_precision", 0.0))
    min_recall = float(class_profile.get("min_recall", 0.0))
    precision_beta = float(class_profile.get("precision_beta", 1.0))
    classification_config = ClassificationConfig(
        n_estimators=n_estimators,
        base_threshold=base_threshold,
        threshold_offset=threshold_offset,
        min_precision=min_precision,
        min_recall=min_recall,
        precision_beta=precision_beta,
    )

    filters_data = data.get("filters", {})
    filter_profiles_raw = filters_data.get("profiles", {})
    if not filter_profiles_raw:
        raise PipelineConfigError("No filter profiles defined in configuration.")

    filter_profile_name = dataset.get("profile", filters_data.get("default_profile"))
    if not filter_profile_name:
        raise PipelineConfigError(
            "No filter profile selected. "
            "Set filters.default_profile or add 'profile' to the dataset entry."
        )
    if filter_profile_name not in filter_profiles_raw:
        available = ", ".join(sorted(filter_profiles_raw))
        raise PipelineConfigError(
            f"Filter profile '{filter_profile_name}' not found. Available profiles: {available}"
        )
    filter_profile = _parse_filter_profile(
        filter_profile_name, filter_profiles_raw[filter_profile_name]
    )

    dataset_a_path = dataset.get("dataset_a")
    dataset_b_path = dataset.get("dataset_b")
    truth_path = dataset.get("truth")
    for label, path in (("dataset_a", dataset_a_path), ("dataset_b", dataset_b_path), ("truth", truth_path)):
        if not path:
            raise PipelineConfigError(
                f"Dataset '{dataset_key}' is missing required path '{label}'."
            )
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.normpath(os.path.join(config_dir, expanded))
        else:
            expanded = os.path.normpath(expanded)
        if expanded != path:
            LOGGER.debug("Expanded %s path from '%s' to '%s'", label, path, expanded)
        if label == "dataset_a":
            dataset_a_path = expanded
        elif label == "dataset_b":
            dataset_b_path = expanded
        else:
            truth_path = expanded

    output_csv_expanded = os.path.expanduser(output_csv)
    if not os.path.isabs(output_csv_expanded):
        output_csv = os.path.normpath(os.path.join(config_dir, output_csv_expanded))
    else:
        output_csv = os.path.normpath(output_csv_expanded)

    return PipelineConfig(
        dataset_key=dataset_key,
        dataset_a=dataset_a_path,
        dataset_b=dataset_b_path,
        truth=truth_path,
        id_column=id_column,
        attributes=list(attribute_list),
        blocking=blocking_config,
        comparisons=comparison_pairs,
        filters=filter_profile,
        classification=classification_config,
        output_csv=output_csv,
        use_gpu=use_gpu,
    )


def list_available_datasets(config_path: str) -> List[str]:
    """Return the dataset keys defined in the configuration file."""

    data = _load_toml(config_path)
    datasets = data.get("datasets", [])
    keys: List[str] = []
    for entry in datasets:
        key = entry.get("key")
        if key:
            keys.append(key)
    return keys


__all__ = [
    "BlockingANNConfig",
    "BlockingConfig",
    "ClassificationConfig",
    "ComparisonPairConfig",
    "ConditionConfig",
    "FilterGroupConfig",
    "FilterProfileConfig",
    "PipelineConfig",
    "PipelineConfigError",
    "list_available_datasets",
    "load_pipeline_config",
]
