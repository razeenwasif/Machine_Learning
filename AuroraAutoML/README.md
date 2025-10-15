# AuroraAutoML

AuroraAutoML is a GPU-aware AutoML pipeline that delivers:

- Loading heterogeneous tabular datasets (CSV, JSON lines, Parquet).
- Performing automated preprocessing and feature engineering.
- Running exploratory data analysis with dataset quality checks.
- Applying conservative automated data cleaning when issues are detected.
- Automatically selecting between regression, classification, clustering, or feedforward neural networks based on the task.
- Running fast GPU-accelerated training with PyTorch.
- Executing hyperparameter optimisation (HPO) with Optuna to locate the best-performing configuration.
- Managing GPU memory proactively with manual garbage collection and cache flushing.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --data path/to/data.csv --target target_column
```

Launch the interactive AuroraAutoML dashboard:

```bash
streamlit run src/gui/app.py
```

The dashboard lets you upload or point to a dataset, run the AutoML pipeline, and inspect data quality diagnostics, automatic cleaning steps, and model performance visualisations.

Key options:

- `--data`: path to dataset (CSV, JSON, JSONL, Parquet).
- `--target`: optional target column for supervised tasks. Skip to trigger clustering.
- `--max-trials`: number of hyperparameter optimisation trials (default: 20).
- `--task`: override automatic task detection (`regression`, `classification`, `clustering`, `auto`).

## Project Layout

```
src/
├── data/
│    └── loaders.py          # Flexible data ingestion
├── models/
│    ├── base.py             # Common model abstraction
│    ├── clustering.py       # GPU K-Means
 │    ├── linear_regression.py
 │    ├── logistic_regression.py
 │    ├── model_factory.py    # Auto model selection
 │    └── neural_network.py
 ├── utils/
│    ├── device.py           # GPU device helpers
│    ├── metrics.py          # Adaptive metric selection
│    └── preprocessing.py    # Feature engineering
├── analysis/
│    ├── analyzer.py         # Exploratory analysis summaries
│    ├── cleaner.py          # Automatic data cleaning
│    └── report.py           # Dataclasses shared across components
├── gui/
│    └── app.py              # Streamlit dashboard entrypoint
├── hpo.py                   # Optuna-based optimisation
├── pipeline.py              # End-to-end orchestration
└── main.py                  # CLI entry point
```

## Reproducible Runs

Set the `SEED` environment variable to fix random seeds. You can also toggle deterministic behaviour via `--deterministic` at the CLI, which activates PyTorch CUDA determinism (with a potential speed cost).

## GPU Memory Safety

The training loop uses manual garbage collection (`gc.collect()`) and `torch.cuda.empty_cache()` between trials to avoid OOMs on smaller GPUs.

## Extensibility

- Add new loaders by extending `BaseDataLoader`.
- Add models by subclassing `BaseModel` and registering in `ModelFactory`.
- Plug in additional Optuna objective components (latency, energy, etc.).

Please consult `python -m src.main --help` for the most up-to-date CLI options.
