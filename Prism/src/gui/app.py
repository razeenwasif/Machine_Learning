"""Streamlit dashboard for interactive exploration of the AutoML pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from recordLinkage.src import PipelineConfigError, list_available_datasets

from src.data.loaders import UnsupportedFormatError, load_dataset
from src.linkage import RecordLinkageDependencyError, RecordLinkagePipeline, RecordLinkageResult
from src.pipeline import AutoMLPipeline, PipelineResult


st.set_page_config(page_title="GPU AutoML Dashboard", layout="wide")


try:
    DEFAULT_LINKAGE_CONFIG = str(RecordLinkagePipeline().config_path)
except FileNotFoundError:
    DEFAULT_LINKAGE_CONFIG = ""


@st.cache_data(show_spinner=False)
def _get_linkage_dataset_keys(config_path: str) -> list[str]:
    if not config_path:
        return []
    try:
        return list_available_datasets(config_path)
    except Exception:  # noqa: BLE001 - configuration issues propagated to UI elsewhere
        return []


def _chown_if_requested(path: Path) -> None:
    """Adjust ownership to match the host user if PRISM_UID/GID are exported."""
    if not path.exists():
        return
    uid_env = os.environ.get("PRISM_UID")
    gid_env = os.environ.get("PRISM_GID")
    if not uid_env or not gid_env:
        return
    try:
        uid = int(uid_env)
        gid = int(gid_env)
    except ValueError:
        return
    try:
        os.chown(path, uid, gid)
    except PermissionError:
        # We are running as an unprivileged user inside the container.
        return
    except OSError:
        # Path may no longer exist or filesystem may not allow chown.
        return


def _persist_uploaded_file(upload) -> Path:
    suffix = Path(upload.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        persisted_path = Path(tmp.name)
    _chown_if_requested(persisted_path)
    return persisted_path


@st.cache_data(show_spinner=False)
def _load_preview(path: str) -> pd.DataFrame:
    bundle = load_dataset(path)
    return bundle.frame.head(500)  # limit preview volume for responsiveness


def _render_metrics(result: PipelineResult) -> None:
    st.subheader("Model Performance")
    metrics_items = [
        {"metric": metric, "value": value}
        for metric, value in result.metrics.items()
    ]
    if metrics_items:
        metrics_df = pd.DataFrame(metrics_items)
        fig = px.bar(
            metrics_df,
            x="metric",
            y="value",
            title="Evaluation Metrics",
            text_auto=".3f",
        )
        fig.update_layout(xaxis_title="", yaxis_title="", bargap=0.3)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation metrics were produced.")


def _render_analysis(result: PipelineResult, sample_df: Optional[pd.DataFrame]) -> None:
    report = result.analysis_report

    st.subheader("Data Overview")
    preview_df = pd.DataFrame(report.original_preview)
    if not preview_df.empty:
        st.dataframe(preview_df, use_container_width=True)
    else:
        st.write("Preview unavailable – dataset may be very large or empty.")

    missing_df = pd.DataFrame(
        {
            "column": list(report.missing_by_column.keys()),
            "missing_fraction": list(report.missing_by_column.values()),
        }
    )
    if not missing_df.empty and missing_df["missing_fraction"].gt(0).any():
        fig = px.bar(
            missing_df.sort_values("missing_fraction", ascending=False),
            x="column",
            y="missing_fraction",
            title="Missing Values by Column",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    if report.notes:
        st.warning(" | ".join(report.notes))

    numeric_profiles = [profile.to_dict() for profile in report.numeric_profiles]
    categorical_profiles = [profile.to_dict() for profile in report.categorical_profiles]

    if numeric_profiles:
        st.subheader("Numeric Summary")
        numeric_df = pd.DataFrame(numeric_profiles)
        table_col, chart_col = st.columns([2, 3])
        table_col.dataframe(numeric_df, use_container_width=True)

        numeric_columns = [
            profile["name"]
            for profile in numeric_profiles
            if sample_df is not None
            and profile["name"] in sample_df.columns
            and pd.api.types.is_numeric_dtype(sample_df[profile["name"]])
        ]
        if numeric_columns:
            default_selection = numeric_columns[: min(3, len(numeric_columns))]
            selected_columns = chart_col.multiselect(
                "Select numeric columns to visualise",
                options=numeric_columns,
                default=default_selection,
            )
            if selected_columns:
                tabs = chart_col.tabs([col for col in selected_columns])
                for tab, column in zip(tabs, selected_columns):
                    column_data = sample_df[column].dropna()
                    if column_data.empty:
                        tab.info("Insufficient data for visualisation.")
                        continue
                    hist_fig = px.histogram(
                        x=column_data,
                        nbins=min(40, max(10, len(column_data) // 5)),
                        title=f"Distribution of {column}",
                    )
                    hist_fig.update_xaxes(title=column)
                    hist_fig.update_yaxes(title="Frequency")
                    tab.plotly_chart(hist_fig, use_container_width=True)

                    box_fig = px.box(
                        sample_df,
                        y=column,
                        points="outliers",
                        title=f"Box Plot of {column}",
                    )
                    box_fig.update_yaxes(title=column)
                    tab.plotly_chart(box_fig, use_container_width=True)
            else:
                chart_col.info("Select one or more columns to generate charts.")
        else:
            chart_col.info("Charts unavailable – numeric columns were not found in the sample.")

    if categorical_profiles:
        st.subheader("Categorical Snapshot")
        categorical_df = pd.DataFrame(categorical_profiles)
        categorical_df["top_frequencies"] = categorical_df["top_frequencies"].apply(
            lambda values: ", ".join(f"{label}: {count}" for label, count in values)
        )
        st.dataframe(categorical_df, use_container_width=True)

    if report.target_summary:
        st.subheader("Target Distribution")
        target_summary = report.target_summary.to_dict()
        if target_summary["type"] == "categorical":
            target_df = pd.DataFrame(
                {
                    "category": list(target_summary["distribution"].keys()),
                    "fraction": list(target_summary["distribution"].values()),
                }
            )
            fig = px.bar(target_df, x="category", y="fraction", title="Target Balance")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            stats_df = pd.DataFrame(
                target_summary["distribution"].items(), columns=["Statistic", "Value"]
            )
            st.table(stats_df)

    if report.correlation_pairs:
        st.subheader("Top Feature Correlations")
        corr_df = pd.DataFrame([pair.to_dict() for pair in report.correlation_pairs])
        st.dataframe(corr_df, use_container_width=True)


def _render_cleaning(result: PipelineResult) -> None:
    cleaning = result.cleaning_report
    st.subheader("Automated Cleaning Steps")
    if not cleaning.has_changes():
        st.info("No cleaning actions were required.")
        return
    if cleaning.applied_steps:
        for step in cleaning.applied_steps:
            st.write(f"- {step}")
    if cleaning.dropped_columns:
        st.write("Dropped columns:")
        st.code(", ".join(cleaning.dropped_columns))
    if cleaning.filled_columns:
        filled_df = pd.DataFrame(
            [{"column": column, "strategy": strategy} for column, strategy in cleaning.filled_columns.items()]
        )
        st.write("Missing value strategies")
        st.table(filled_df)
    if cleaning.outlier_treatments:
        outlier_df = pd.DataFrame(
            [{"column": column, "action": action} for column, action in cleaning.outlier_treatments.items()]
        )
        st.write("Outlier handling")
        st.table(outlier_df)


def _resolve_id_column(frame: pd.DataFrame, id_column: str) -> str:
    if id_column in frame.columns:
        return id_column
    lower_map = {column.lower(): column for column in frame.columns}
    match = lower_map.get(id_column.lower())
    if match:
        return match
    raise KeyError(f"ID column '{id_column}' not found in dataset columns: {frame.columns.tolist()}")


def _resolve_attribute_columns(frame: pd.DataFrame, attributes: list[str], id_column: str) -> list[str]:
    lower_map = {column.lower(): column for column in frame.columns}
    resolved: list[str] = []
    for attr in attributes:
        candidate = lower_map.get(attr.lower())
        if candidate and candidate != id_column and candidate not in resolved:
            resolved.append(candidate)
    return resolved


def _build_linked_dataset(result: RecordLinkageResult) -> tuple[Path, pd.DataFrame]:
    matches_df = pd.read_csv(result.output_path, header=None, names=["rec_id_A", "rec_id_B"])

    dataset_a = load_dataset(str(result.dataset_a_path)).frame
    dataset_b = load_dataset(str(result.dataset_b_path)).frame

    id_col_a = _resolve_id_column(dataset_a, result.id_column)
    id_col_b = _resolve_id_column(dataset_b, result.id_column)

    attr_cols_a = _resolve_attribute_columns(dataset_a, result.attributes, id_col_a)
    attr_cols_b = _resolve_attribute_columns(dataset_b, result.attributes, id_col_b)

    keep_cols_a = [id_col_a, *attr_cols_a] if attr_cols_a else dataset_a.columns.tolist()
    keep_cols_b = [id_col_b, *attr_cols_b] if attr_cols_b else dataset_b.columns.tolist()
    dataset_a = dataset_a[keep_cols_a]
    dataset_b = dataset_b[keep_cols_b]

    df_a = dataset_a.rename(columns={id_col_a: "rec_id_A"})
    df_b = dataset_b.rename(columns={id_col_b: "rec_id_B"})

    rename_a = {column: f"A_{column}" for column in df_a.columns if column != "rec_id_A"}
    rename_b = {column: f"B_{column}" for column in df_b.columns if column != "rec_id_B"}
    df_a = df_a.rename(columns=rename_a)
    df_b = df_b.rename(columns=rename_b)

    linked_df = matches_df.merge(df_a, on="rec_id_A", how="left").merge(df_b, on="rec_id_B", how="left")
    if "is_match" not in linked_df.columns:
        linked_df.insert(0, "is_match", 1)

    tmp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name)
    linked_df.to_csv(tmp_path, index=False)
    _chown_if_requested(tmp_path)
    preview_df = linked_df.head(200)
    return tmp_path, preview_df


def _use_linked_dataset_in_automl() -> None:
    path = st.session_state.get("linked_dataset_path")
    if not path:
        return
    st.session_state["data_source_choice"] = "Path"
    st.session_state["dataset_path_input"] = path
    st.session_state["target_column_input"] = "is_match"
    st.session_state["uploaded_dataset_path"] = ""
    st.session_state["uploaded_dataset_name"] = ""
    st.session_state["uploaded_dataset_size"] = 0


def _resolve_dataset_path(source: str, upload) -> Optional[str]:
    if source == "Path":
        return st.sidebar.text_input("Dataset path", key="dataset_path_input")
    if source == "Upload":
        if upload is None:
            return st.session_state.get("uploaded_dataset_path", "")
        existing_name = st.session_state.get("uploaded_dataset_name")
        existing_size = st.session_state.get("uploaded_dataset_size")
        if existing_name == upload.name and existing_size == upload.size:
            return st.session_state.get("uploaded_dataset_path", "")
        stored_path = _persist_uploaded_file(upload)
        st.session_state["uploaded_dataset_path"] = str(stored_path)
        st.session_state["uploaded_dataset_name"] = upload.name
        st.session_state["uploaded_dataset_size"] = upload.size
        return str(stored_path)
    return ""


def main() -> None:
    st.title("GPU AutoML Dashboard")
    st.caption("Run automated data analysis, cleaning, and model selection with visual feedback.")

    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None
    if "data_source_choice" not in st.session_state:
        st.session_state["data_source_choice"] = "Path"
    if "dataset_path_input" not in st.session_state:
        st.session_state["dataset_path_input"] = ""
    if "uploaded_dataset_path" not in st.session_state:
        st.session_state["uploaded_dataset_path"] = ""
    if "uploaded_dataset_name" not in st.session_state:
        st.session_state["uploaded_dataset_name"] = ""
    if "uploaded_dataset_size" not in st.session_state:
        st.session_state["uploaded_dataset_size"] = 0
    if "preview_df" not in st.session_state:
        st.session_state["preview_df"] = None
    if "target_column_input" not in st.session_state:
        st.session_state["target_column_input"] = ""
    if "linkage_result" not in st.session_state:
        st.session_state["linkage_result"] = None
    if "linked_dataset_path" not in st.session_state:
        st.session_state["linked_dataset_path"] = ""
    if "linked_preview_df" not in st.session_state:
        st.session_state["linked_preview_df"] = None
    if "linkage_config_path" not in st.session_state:
        st.session_state["linkage_config_path"] = DEFAULT_LINKAGE_CONFIG
    if "linkage_output_override" not in st.session_state:
        st.session_state["linkage_output_override"] = ""
    if "latest_linkage_output" not in st.session_state:
        st.session_state["latest_linkage_output"] = ""
    if "linkage_use_gpu" not in st.session_state:
        st.session_state["linkage_use_gpu"] = True
    if "linkage_skip_filters" not in st.session_state:
        st.session_state["linkage_skip_filters"] = False

    sidebar = st.sidebar
    sidebar.header("Configuration")

    data_source = sidebar.radio(
        "Data source",
        ["Path", "Upload"],
        horizontal=True,
        key="data_source_choice",
    )
    uploaded_data = None
    if data_source == "Upload":
        uploaded_data = sidebar.file_uploader(
            "Upload dataset",
            type=["csv", "json", "jsonl", "parquet", "tsv", "txt"],
        )

    dataset_path = _resolve_dataset_path(data_source, uploaded_data)
    target_column = sidebar.text_input("Target column (optional)", key="target_column_input")
    task = sidebar.selectbox(
        "Task override",
        options=["auto", "regression", "classification", "clustering"],
        index=0,
    )
    test_size = sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    max_trials = sidebar.slider("Max trials", min_value=5, max_value=50, value=20, step=5)
    seed = sidebar.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42, step=1)
    deterministic = sidebar.checkbox("Deterministic mode", value=False)
    prefer_gpu = sidebar.checkbox("Use GPU when available", value=True)

    linkage_expander = sidebar.expander("Record linkage (optional)")
    with linkage_expander:
        config_path = st.text_input("Config path", key="linkage_config_path")
        dataset_keys = _get_linkage_dataset_keys(config_path)
        dataset_key: Optional[str] = None
        if dataset_keys:
            current_key = st.session_state.get("linkage_dataset_key")
            if current_key not in dataset_keys:
                st.session_state["linkage_dataset_key"] = dataset_keys[0]
            dataset_key = st.selectbox("Dataset preset", options=dataset_keys, key="linkage_dataset_key")
        else:
            st.info("No dataset presets found. Enter a dataset key manually.")
            dataset_key = st.text_input("Dataset key", key="linkage_dataset_key_input")
        output_override = st.text_input(
            "Output CSV override (optional)",
            key="linkage_output_override",
            placeholder="Leave blank to use config value",
        )
        use_gpu_override = st.checkbox("Force GPU comparisons", key="linkage_use_gpu")
        skip_filters = st.checkbox("Skip precision filters", key="linkage_skip_filters")
        run_linkage = st.button("Run record linkage", use_container_width=True)

        if run_linkage:
            if not config_path:
                st.error("Provide a configuration path before running record linkage.")
            elif not dataset_key:
                st.error("Select or enter a dataset preset before running record linkage.")
            else:
                try:
                    pipeline = RecordLinkagePipeline(Path(config_path))
                except FileNotFoundError as exc:
                    st.error(str(exc))
                else:
                    with st.spinner("Running record linkage pipeline..."):
                        try:
                            output_path_argument = (
                                Path(output_override).expanduser()
                                if output_override
                                else None
                            )
                            result = pipeline.run(
                                dataset_key=dataset_key or None,
                                output_path=output_path_argument,
                                use_gpu=use_gpu_override,
                                skip_filters=skip_filters,
                            )
                            _chown_if_requested(Path(result.output_path))
                            linked_path, linked_preview = _build_linked_dataset(result)
                            st.session_state["linkage_result"] = result
                            st.session_state["linked_dataset_path"] = str(linked_path)
                            st.session_state["linked_preview_df"] = linked_preview
                            st.session_state["latest_linkage_output"] = str(result.output_path)
                            st.success(
                                f"Matched {result.match_count:,} pairs (saved to {result.output_path})."
                            )
                        except RecordLinkageDependencyError as exc:
                            st.session_state["linkage_result"] = None
                            st.error(f"Record linkage unavailable: {exc}")
                        except PipelineConfigError as exc:
                            st.session_state["linkage_result"] = None
                            st.error(f"Configuration error: {exc}")
                        except Exception as exc:  # noqa: BLE001
                            st.session_state["linkage_result"] = None
                            st.error(f"Record linkage failed: {exc}")

        linkage_result = st.session_state.get("linkage_result")
        if linkage_result:
            st.caption(f"Last run preset: {linkage_result.dataset_key}")
            metrics_df = pd.DataFrame(
                [{"metric": key.title().replace("_", " "), "value": value} for key, value in linkage_result.linkage_metrics.items()]
            )
            if not metrics_df.empty:
                st.write("Linkage quality")
                st.table(metrics_df.style.format({"value": "{:.4f}"}))

            blocking_df = pd.DataFrame(
                [{"metric": key.replace("_", " ").title(), "value": value} for key, value in linkage_result.blocking_metrics.items()]
            )
            if not blocking_df.empty:
                st.write("Blocking summary")
                st.table(blocking_df)

            runtime_df = pd.DataFrame(
                [{"stage": name.replace("_", " ").title().replace("Seconds", ""), "seconds": value}
                 for name, value in linkage_result.runtime.items()]
            )
            if not runtime_df.empty:
                st.write("Runtime breakdown (s)")
                st.table(runtime_df.style.format({"seconds": "{:.3f}"}))

            linked_preview = st.session_state.get("linked_preview_df")
            if linked_preview is not None and not linked_preview.empty:
                st.write("Linked dataset preview")
                st.dataframe(linked_preview, use_container_width=True)

            matches_path = Path(linkage_result.output_path)
            if matches_path.exists():
                st.download_button(
                    "Download matched pairs CSV",
                    data=matches_path.read_bytes(),
                    file_name=matches_path.name,
                    mime="text/csv",
                )

            linked_path_str = st.session_state.get("linked_dataset_path")
            if linked_path_str:
                linked_path_obj = Path(linked_path_str)
                if linked_path_obj.exists():
                    st.download_button(
                        "Download linked feature dataset",
                        data=linked_path_obj.read_bytes(),
                        file_name=linked_path_obj.name,
                        mime="text/csv",
                    )
                    st.button(
                        "Use linked dataset in AutoML",
                        use_container_width=True,
                        on_click=_use_linked_dataset_in_automl,
                    )

    run_pipeline = sidebar.button("Run analysis & training", use_container_width=True)

    preview_container = st.container()
    if dataset_path:
        if dataset_path == st.session_state.get("linked_dataset_path"):
            st.info("Using linked dataset generated from the record linkage step.")
        try:
            preview_df = _load_preview(dataset_path)
            st.session_state["preview_df"] = preview_df
            with preview_container:
                st.markdown("#### Dataset Preview")
                st.dataframe(preview_df, use_container_width=True)
        except (FileNotFoundError, UnsupportedFormatError) as exc:
            st.error(str(exc))
            st.session_state["preview_df"] = None
        except ValueError as exc:
            st.error(f"Unable to load dataset: {exc}")
            st.session_state["preview_df"] = None
    else:
        st.session_state["preview_df"] = None

    if run_pipeline:
        if not dataset_path:
            st.error("Please provide a dataset path or upload a file before running the pipeline.")
        else:
            with st.spinner("Running AutoML pipeline..."):
                try:
                    pipeline = AutoMLPipeline(
                        seed=int(seed),
                        deterministic=deterministic,
                        max_trials=int(max_trials),
                        prefer_gpu=prefer_gpu,
                    )
                    result = pipeline.run(
                        data_path=dataset_path,
                        target=target_column or None,
                        task=task,
                        test_size=float(test_size),
                        max_trials=int(max_trials),
                    )
                    st.session_state["pipeline_result"] = result
                    st.success(f"Best model: {result.model_name}")
                except Exception as exc:  # noqa: BLE001 - surface any pipeline issues
                    st.session_state["pipeline_result"] = None
                    st.error(f"Pipeline failed: {exc}")

    result: Optional[PipelineResult] = st.session_state.get("pipeline_result")
    if result:
        st.divider()
        _render_metrics(result)
        st.divider()
        _render_analysis(result, st.session_state.get("preview_df"))
        st.divider()
        _render_cleaning(result)


if __name__ == "__main__":
    main()
