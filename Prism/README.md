# Prism

> GPU-aware automated machine learning for heterogeneous tabular datasets with built-in EDA, cleaning, and hyperparameter search.

Prism blends deterministic data preparation, GPU-first training, and Optuna-powered search into a single reproducible pipeline. It can be driven entirely from the command line or through a Streamlit dashboard that surfaces analysis artefacts and performance insights.

## Table of Contents
- [Key Capabilities](#key-capabilities)
- [Core Architecture](#core-architecture)
- [Libraries & Frameworks](#libraries--frameworks)
- [Installation](#installation)
- [Command Line Quick Start](#command-line-quick-start)
- [Record Linkage Quick Start](#record-linkage-quick-start)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Data Requirements & Recommendations](#data-requirements--recommendations)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Configuration Reference](#configuration-reference)
- [Outputs & Artefacts](#outputs--artefacts)
- [Extending Prism](#extending-prism)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [Project Layout](#project-layout)

## Key Capabilities
- End-to-end AutoML for regression, classification, and clustering with optional automatic task detection.
- Robust ingestion of CSV, TSV, JSON/JSONL, and Parquet files with schema introspection.
- Comprehensive exploratory data analysis (EDA) reporting, including missingness, distribution summaries, and correlation mining.
- Conservative automated cleaning: duplicate removal, high-missing column drops, imputation, and outlier clipping.
- Feature engineering powered by `ColumnTransformer`; consistent numeric scaling and categorical one-hot encoding.
- GPU-first model zoo (linear/logistic regression, configurable feedforward nets, GPU K-Means) with fallbacks to CPU/MPS.
- GPU-accelerated record linkage workflow for entity resolution with configurable blocking, similarity, filtering, and supervised classification stages.
- Optuna-backed hyperparameter optimisation with per-model search spaces and early stopping.
- Memory-aware runtime that empties CUDA caches between trials to reduce OOM risk on constrained hardware.
- Rich CLI summaries and optional JSON exports for downstream automation.
- Streamlit dashboard with interactive previews, charts, cleaning logs, and metric visualisations.

## Core Architecture
```
             ┌─────────────────────────────────────────────────────┐
             │                   AutoMLPipeline                    │
             └──────────────┬──────────────┬──────────────┬────────┘
                            │              │              │
                   Data Ingestion      Analysis &      Model Search
                       (loaders)        Cleaning       & Training
                        │              (analysis/        (models,
                        ▼          cleaner + reports)      hpo)
                 DatasetBundle              │              │
                        │                   ▼              │
                        └─────▶ Preprocessor ─────▶ Feature tensors
                                              │
                              Evaluation & Metric Suite
                                              │
                               PipelineResult aggregation
```

At execution time the CLI/GUI instantiate `AutoMLPipeline`, which orchestrates:

1. **Dataset ingestion** via format-aware loaders.
2. **EDA & quality checks** with `DataAnalyzer`.
3. **Cleaning** using `DataCleaner`, guided by the analysis report.
4. **Feature engineering** with the reusable `Preprocessor`.
5. **Candidate generation** via `models.model_factory`.
6. **Optuna optimisation** wrapped by `HyperparameterOptimizer`.
7. **Final training & evaluation** returning a `PipelineResult` for presentation or export.

## Libraries & Frameworks
Prism is powered by the following ecosystem:

| Library | Purpose |
| --- | --- |
| `torch` | GPU-accelerated tensor operations and deep learning primitives. |
| `optuna` | Bayesian hyperparameter optimisation with study management. |
| `pandas` | DataFrame ingestion, cleaning, and previewing. |
| `numpy` | Numerical utilities, particularly during preprocessing and metrics. |
| `scikit-learn` | ColumnTransformer pipelines, preprocessing stages, and evaluation metrics. |
| `pyarrow` | Parquet ingestion backend for Arrow-based file formats. |
| `rich` | Rich-text tables and logging for the CLI experience. |
| `streamlit` | Web dashboard for interactive execution and visualisation. |
| `plotly` | Dashboard charts for metrics, distributions, and missing data. |

## Installation
Prism targets Python 3.10+ and optionally leverages NVIDIA GPUs when available.

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### GPU notes
- For CUDA acceleration install the appropriate PyTorch build: `pip install torch --index-url https://download.pytorch.org/whl/cu118`.
- Apple Silicon users can rely on the MPS backend automatically.
- Pass `--no-gpu` to force CPU execution if CUDA drivers are unavailable or unstable.

### Record linkage dependencies
The record linkage workflow depends on the RAPIDS stack (`cudf`, `cuml`, `cupy`, `faiss`, `numba`, `rapidfuzz`). Use the provided `rapids-rl.yml` to create the compatible environment:

```bash
conda env create -f rapids-rl.yml
conda activate rapids-rl
```

This pins Python 3.10, RAPIDS 23.06 (CUDA 11.8), and matching dependencies. The AutoML pipeline does not require these packages and can be installed with only `requirements.txt` when record linkage is not needed.

## Command Line Quick Start
Run a full AutoML cycle against a dataset:

```bash
python -m src.main \
  --data datasets/customer_churn.csv \
  --target churned \
  --task auto \
  --max-trials 25 \
  --output-json results.json
```

The CLI prints:
- Dataset shape and any automated cleaning actions.
- Best model name, task type, and Optuna-derived configuration.
- Evaluation metrics rendered in a `rich` table.
- Target distribution and top correlations when applicable.
- A JSON artefact (when `--output-json` is provided) containing the aggregated reports.

View all options:

```bash
python -m src.main --help
```

## Record Linkage Quick Start
Prism bundles the GPU entity resolution pipeline from `recordLinkage/`. This mode analyses both input datasets with Prism's EDA tooling, then runs blocking, similarity, filtering, and supervised classification to generate matched pairs.

List the available dataset presets defined in the TOML configuration:

```bash
python -m src.main link --list-datasets
```

Run a preset (the default configuration lives at `recordLinkage/config/pipeline.toml`):

```bash
python -m src.main link --dataset assignment_datasets
```

Useful flags:
- `--config path/to/pipeline.toml` to point at an alternate configuration file.
- `--output ./out/matches.csv` to override the output CSV path for the current run.
- `--skip-filters` to disable the high-precision filters while tuning thresholds.
- `--use-gpu` / `--no-gpu` to override the config default for similarity comparisons.

The CLI prints dataset shapes, notable data-quality notes, blocking metrics, linkage quality (precision/recall/F1/accuracy), and stage runtimes. Matched pairs are written to the configured CSV.

Prefer a GUI? Open `streamlit run src/gui/app.py` and use the “Record linkage (optional)” panel to execute a preset, preview the fused dataset, download the results, and push the linked records directly into the AutoML workflow. The panel uses the same RAPIDS environment as the CLI, so activate `rapids-rl` before launching Streamlit when you plan to run linkage.

## Streamlit Dashboard
Launch the interactive UI for dataset uploads, parameter tuning, and charting:

```bash
streamlit run src/gui/app.py
```

Dashboard highlights:
- Upload files or reference filesystem paths.
- Configure task overrides, test split size, trial budgets, seeds, and determinism.
- Preview the dataset (first 500 rows) with automatic caching for responsiveness.
- Observe metric bar charts, distribution histograms, target balance, and correlation tables.
- Inspect the cleaning play-by-play (dropped columns, filled values, outlier treatments).

## Data Requirements & Recommendations
- Input formats: `.csv`, `.tsv`, `.txt`, `.json`, `.jsonl`, `.parquet`.
- Column headers are required for correct feature detection.
- For supervised tasks provide `--target`. Omit it to trigger clustering mode.
- Ensure categorical targets do not exceed the available memory when one-hot encoded.
- Large datasets benefit from GPU acceleration; consider reducing `--max-trials` on constrained hardware.

## Pipeline Walkthrough
Prism follows a deterministic chain of stages:

1. **Load** (`src/data/loaders.py`): File-type detection selects the correct loader, validates existence, and guarantees a non-empty DataFrame.
2. **Analyse** (`src/analysis/analyzer.py`): Produces missing-value maps, numeric/categorical profiles, correlation pairs, duplicates, and target imbalance notes.
3. **Clean** (`src/analysis/cleaner.py`): Drops high-missing columns, deduplicates rows, imputes remaining missing values, and clips outliers using the IQR rule.
4. **Infer task** (`AutoMLPipeline._infer_task`): Chooses regression, classification, or clustering based on target dtype and cardinality when `--task auto` is used.
5. **Split data**: Stratified splits for classification; random splits for other tasks with reproducible seeding.
6. **Preprocess** (`src/utils/preprocessing.py`):
   - Numeric columns → median imputation + standard scaling.
   - Categorical columns → most-frequent imputation + one-hot encoding.
   - Targets → ordinal encoding for classification or float tensors for regression.
7. **Hyperparameter optimisation** (`src/hpo.py`):
   - Candidate models enumerated by `models.model_factory.candidates_for_task`.
   - Optuna samples model-specific search spaces (learning rates, epochs, layer widths, cluster counts).
   - Validation splits evaluate each trial with task-aware metrics.
   - GPU cache is flushed between trials to prevent fragmentation.
8. **Final training & evaluation**:
   - Best trial configuration is retrained on the full training set.
   - Metrics (`src/utils/metrics.py`) compute RMSE/MAE/R², accuracy/F1, or silhouette/Calinski-Harabasz.
9. **Reporting**: All artefacts (analysis, cleaning, metrics, config) are assembled into `PipelineResult` for downstream consumption.

## Configuration Reference
- `--data` (required): Path to dataset file.
- `--target`: Target column name for supervised learning.
- `--task`: Override task detection (`auto`, `regression`, `classification`, `clustering`).
- `--max-trials`: Optuna trial budget (default `20`).
- `--test-size`: Hold-out ratio for final evaluation (default `0.2`).
- `--seed`: Global random seed for numpy, torch, and Optuna.
- `--deterministic`: Enforce deterministic CUDA kernels (`torch.backends.cudnn.deterministic=True`).
- `--no-gpu`: Force CPU execution even if CUDA/MPS is available.
- `--output-json`: File path to persist pipeline output as JSON.

Set `SEED=<value>` in the environment to mirror CLI seeding during subprocess launches.

## Outputs & Artefacts
The CLI/GUI surface information derived from `PipelineResult`:

| Field | Description |
| --- | --- |
| `task` | Resolved task type. |
| `model_name` | Candidate identifier (`linear_regression`, `logistic_regression`, `neural_network`, `kmeans`). |
| `metrics` | Dictionary of evaluation metrics keyed by name. |
| `best_config` | Union of base model defaults and Optuna-proposed hyperparameters. |
| `hpo_score` | Best validation score achieved during optimisation. |
| `analysis_report` | Structured EDA output (column stats, missingness, correlations, notes). |
| `cleaning_report` | Applied cleaning steps with dropped columns, imputations, and outlier handling. |

When exporting via `--output-json`, the JSON payload mirrors the dataclasses, making it easy to integrate with monitoring, experiment tracking, or dashboards.

## Extending Prism
- **New data format**: Register a loader in `src/data/loaders.py` and extend `LOADERS`.
- **Additional model**:
  1. Subclass `BaseModel` or `SupervisedTorchModel`.
  2. Register it in `models/model_factory.candidates_for_task`.
  3. Optionally add a search space in `src/hpo.py` via `@register_search_space`.
- **Custom metrics**: Update `src/utils/metrics.py` or supply alternative metric functions inside `AutoMLPipeline._select_primary_metric`.
- **Augmented cleaning**: Enhance `DataCleaner` strategies or thresholds to match domain requirements.

These extension points keep the pipeline cohesive while allowing task-specific experimentation.

## Development Workflow
- Create or activate a virtual environment (see [Installation](#installation)).
- Format and lint using tools of your choice; the codebase is compatible with `ruff` and `black` defaults.
- Manual testing: run `python -m src.main --help` to verify CLI wiring, then execute sample datasets to validate end-to-end behaviour.
- Streamlit development: use `streamlit run src/gui/app.py --server.runOnSave true` for live reload during UI tweaks.

## Troubleshooting
- **Dataset not found**: Verify the supplied path resolves on the local filesystem; absolute paths are recommended.
- **Unsupported file extension**: Convert data to CSV/Parquet/JSONL or register a new loader.
- **Out-of-memory on GPU**: Reduce `--max-trials`, enable `--deterministic`, or pass `--no-gpu` to fall back to CPU.
- **Categorical target errors**: Ensure `--target` references a column present in the dataset and contains repeatable categories across train/test splits.
- **Optuna trial failures**: Failures surface directly in the CLI/GUI; inspect the stack trace and adjust hyperparameter bounds or input data quality.

## Project Layout
```
src/
├── analysis/              # EDA and cleaning reports
├── data/                  # Format-aware dataset loaders
├── gui/                   # Streamlit application
├── models/                # Model zoo and abstractions
├── utils/                 # Device, metrics, preprocessing helpers
├── hpo.py                 # Optuna integration
├── pipeline.py            # AutoML orchestration
└── main.py                # CLI entrypoint
datasets/                  # Example datasets (if provided)
requirements.txt           # Python dependencies
```

Prism is production-ready for scripted automation while remaining flexible for research explorations and rapid prototyping.
