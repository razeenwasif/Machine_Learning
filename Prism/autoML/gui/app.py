"""Streamlit frontend for the Prism microservices."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Add project root to path to allow sibling imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoML.data.loaders import UnsupportedFormatError, load_dataset
from recordLinkage.src import list_available_datasets

# --- API Configuration ---
BACKEND_RL_URL = os.environ.get("BACKEND_RL_URL", "http://backend-rl:8000")
BACKEND_AUTOML_URL = os.environ.get("BACKEND_AUTOML_URL", "http://backend-automl:8001")

st.set_page_config(page_title="Prism Dashboard", layout="wide")

# --- Helper Functions ---

def _chown_if_requested(path: Path) -> None:
    """Adjust ownership to match the host user if PRISM_UID/GID are exported."""
    # This function is kept for scenarios where the frontend might write files
    # that need to be accessed by the user on the host machine.
    if not path.exists():
        return
    uid_env = os.environ.get("PRISM_UID")
    gid_env = os.environ.get("PRISM_GID")
    if not uid_env or not gid_env:
        return
    try:
        os.chown(path, int(uid_env), int(gid_env))
    except (ValueError, PermissionError, OSError):
        pass

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
    return bundle.frame.head(500)

# --- Rendering Functions (adapted for dictionary inputs) ---

def _render_metrics(result: Dict[str, Any]) -> None:
    st.subheader("Model Performance")
    metrics = result.get('metrics', {})
    metrics_items = [{"metric": metric, "value": value} for metric, value in metrics.items()]
    if metrics_items:
        metrics_df = pd.DataFrame(metrics_items)
        fig = px.bar(metrics_df, x="metric", y="value", title="Evaluation Metrics", text_auto=".3f")
        fig.update_layout(xaxis_title="", yaxis_title="", bargap=0.3)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation metrics were produced.")

def _render_analysis(result: Dict[str, Any], sample_df: Optional[pd.DataFrame]) -> None:
    report = result.get('analysis_report', {})
    if not report:
        st.warning("Analysis report not found in the result.")
        return

    st.subheader("Data Overview")
    preview_df = pd.DataFrame(report.get('original_preview', []))
    if not preview_df.empty:
        st.dataframe(preview_df, use_container_width=True)
    else:
        st.write("Preview unavailable.")

    # ... (rest of the analysis rendering can be adapted similarly)


def main() -> None:
    st.title("Prism Dashboard")
    st.caption("Microservices Edition: UI Frontend")

    # Initialize session state comprehensively
    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None
    if "data_source_choice" not in st.session_state:
        st.session_state["data_source_choice"] = "Path"
    if "dataset_path_input" not in st.session_state:
        st.session_state["dataset_path_input"] = ""
    if "uploaded_dataset_path" not in st.session_state:
        st.session_state["uploaded_dataset_path"] = ""
    if "target_column_input" not in st.session_state:
        st.session_state["target_column_input"] = ""
    if "linkage_result" not in st.session_state:
        st.session_state["linkage_result"] = None
    if "linkage_config_path" not in st.session_state:
        st.session_state["linkage_config_path"] = "recordLinkage/config/pipeline.toml"
    if "linkage_dataset_key_input" not in st.session_state:
        st.session_state["linkage_dataset_key_input"] = ""

    sidebar = st.sidebar
    sidebar.header("Configuration")

    # --- Data Source Widgets ---
    data_source = sidebar.radio("Data source", ["Path", "Upload"], horizontal=True, key="data_source_choice")
    uploaded_data = None
    if data_source == "Upload":
        uploaded_data = sidebar.file_uploader("Upload dataset", type=["csv", "json", "jsonl", "parquet", "tsv", "txt"])
    
    dataset_path_input = st.sidebar.text_input("Dataset path", key="dataset_path_input")
    dataset_path = dataset_path_input
    if data_source == "Upload" and uploaded_data:
        persisted_path = _persist_uploaded_file(uploaded_data)
        st.session_state["uploaded_dataset_path"] = str(persisted_path)
        dataset_path = str(persisted_path)

    # --- AutoML Widgets ---
    automl_expander = sidebar.expander("AutoML Pipeline", expanded=True)
    with automl_expander:
        target_column = st.text_input("Target column (optional)", key="target_column_input")
        task = st.selectbox("Task override", options=["auto", "regression", "classification", "clustering"], index=0)
        max_trials = st.slider("Max trials", min_value=5, max_value=50, value=20, step=5)
        run_pipeline = st.button("Run AutoML Pipeline", use_container_width=True)

    # --- Record Linkage Widgets ---
    linkage_expander = sidebar.expander("Record Linkage (optional)")
    with linkage_expander:
        config_path = st.text_input("Config path", key="linkage_config_path", placeholder="e.g., recordLinkage/config/pipeline.toml")
        dataset_key = st.text_input("Dataset key", key="linkage_dataset_key_input")
        run_linkage = st.button("Run Record Linkage", use_container_width=True)

    # --- Pipeline Execution Logic ---
    if run_linkage:
        if not dataset_key:
            st.error("Please provide a dataset key for record linkage.")
        else:
            with st.spinner("Sending request to Record Linkage backend..."):
                payload = {
                    "dataset_key": dataset_key,
                    "config_path": config_path or None,
                }
                try:
                    response = requests.post(f"{BACKEND_RL_URL}/run", json=payload, timeout=900)
                    if response.status_code == 200:
                        st.session_state["linkage_result"] = response.json()
                        st.success("Record linkage pipeline completed successfully!")
                    else:
                        st.error(f"Record Linkage API Error {response.status_code}: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to Record Linkage backend: {e}")

    if run_pipeline:
        if not dataset_path:
            st.error("Please provide a dataset path or upload a file.")
        else:
            with st.spinner("Sending request to AutoML backend..."):
                payload = {
                    "data_path": dataset_path,
                    "target": target_column or None,
                    "task": task,
                    "max_trials": max_trials,
                }
                try:
                    response = requests.post(f"{BACKEND_AUTOML_URL}/run", json=payload, timeout=900)
                    if response.status_code == 200:
                        st.session_state["pipeline_result"] = response.json()
                        st.success("AutoML pipeline completed successfully!")
                    else:
                        st.error(f"AutoML API Error {response.status_code}: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to AutoML backend: {e}")

    # --- Display Results ---
    if st.session_state.get("linkage_result"):
        st.divider()
        st.header("Record Linkage Result")
        st.json(st.session_state["linkage_result"])

    if st.session_state.get("pipeline_result"):
        st.divider()
        st.header("AutoML Result")
        result_data = st.session_state["pipeline_result"]
        st.write(f"**Best Model:** {result_data.get('model_name')}")
        _render_metrics(result_data)
        # _render_analysis(result_data, None) # Simplified for now

if __name__ == "__main__":
    main()