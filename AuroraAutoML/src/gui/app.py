"""Streamlit dashboard for interactive exploration of the AutoML pipeline."""

from __future__ import annotations

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

from src.data.loaders import UnsupportedFormatError, load_dataset
from src.pipeline import AutoMLPipeline, PipelineResult


st.set_page_config(page_title="GPU AutoML Dashboard", layout="wide")


def _persist_uploaded_file(upload) -> Path:
    suffix = Path(upload.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        return Path(tmp.name)


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


def _resolve_dataset_path(source: str, upload) -> Optional[str]:
    if source == "Path":
        return st.sidebar.text_input("Dataset path", value="")
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
    if "uploaded_dataset_path" not in st.session_state:
        st.session_state["uploaded_dataset_path"] = ""
    if "uploaded_dataset_name" not in st.session_state:
        st.session_state["uploaded_dataset_name"] = ""
    if "uploaded_dataset_size" not in st.session_state:
        st.session_state["uploaded_dataset_size"] = 0
    if "preview_df" not in st.session_state:
        st.session_state["preview_df"] = None

    sidebar = st.sidebar
    sidebar.header("Configuration")

    data_source = sidebar.radio("Data source", ["Path", "Upload"], horizontal=True)
    uploaded_data = None
    if data_source == "Upload":
        uploaded_data = sidebar.file_uploader(
            "Upload dataset",
            type=["csv", "json", "jsonl", "parquet", "tsv", "txt"],
        )

    dataset_path = _resolve_dataset_path(data_source, uploaded_data)
    target_column = sidebar.text_input("Target column (optional)", value="")
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

    run_pipeline = sidebar.button("Run analysis & training", use_container_width=True)

    preview_container = st.container()
    if dataset_path:
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
