# Record Linkage GPU Pipeline

## Overview
This project implements a GPU-accelerated record linkage workflow that loads two person-level datasets, blocks records to curtail comparisons, computes similarity vectors, classifies matches with a cuML-based random forest, evaluates metrics, and writes the predicted matches to `out/record_linkage_matches.csv`. The entire pipeline is now configured via `config/pipeline.toml`, so you can drop in your datasets, adjust blocking/comparison/classification settings, and run the linker without diving into the source code. The same workflow is exposed through Prism via `python -m autoML.main link --dataset <preset>`.

## Prerequisites
- NVIDIA GPU with a supported CUDA driver (CUDA 12.x as exported in `conda_env.txt`).
- Conda (Miniconda/Anaconda) to manage the RAPIDS stack.
- Datasets formatted as CSV files containing the attributes referenced in `recordLinkage.py`.

### Environment setup
Create the pinned RAPIDS environment with the bundled YAML:

```bash
conda env create -f ../rapids-rl.yml
conda activate rapids-rl
```

This installs CUDA 11.8-compatible builds of cuDF, cuML, FAISS, CuPy, pandas 1.5, pyarrow 11, and the additional Python dependencies (`rapidfuzz`, Optuna, Streamlit, etc.). If you need to customise the stack (different CUDA toolkit or RAPIDS release), start from the compatibility matrix at https://rapids.ai/start.html and adjust the versions in `rapids-rl.yml`.

## Repository layout
- `config/pipeline.toml` – single source of truth for datasets, blocking, comparisons, filters, and classifier settings.
- `src/recordLinkage.py` – CLI entry point orchestrating the six pipeline stages using the declarative config.
- `src/pipeline_config.py` – config parser and validation helpers.
- `src/config.py` – logging defaults, q-gram length, and a global GPU comparison flag (overridden at runtime by the CLI).
- `src/blocking.py` – blocking strategies, including ANN-based candidate generation.
- `src/comparison.py` – similarity functions with GPU and CPU implementations.
- `src/classification.py` – cuML random forest training, threshold selection, and scoring.
- `src/evaluation.py` – blocking and linkage quality metrics.
- `src/saveLinkResult.py` – writes the match set to CSV.
- `src/datasets/` – default CSV inputs supplied for the assignment.
- `out/` – pipeline output directory (the default run writes `record_linkage_matches.csv`).

## Running the pipeline
List the dataset presets available in `config/pipeline.toml`:
```bash
python src/recordLinkage.py --list-datasets
```

Run a specific preset (the default dataset is picked from the config if `--dataset` is omitted):
```bash
python src/recordLinkage.py --dataset assignment_datasets
```

Common CLI overrides:
- `--config path/to/pipeline.toml` – use an alternate configuration file.
- `--output ./out/custom_results.csv` – change the output path without editing the config.
- `--skip-filters` – bypass the precision filters while tuning thresholds.
- `--no-gpu` / `--use-gpu` – force CPU or GPU similarity comparisons.

## Core settings to tweak
### `config/pipeline.toml`
- `[defaults]` – shared settings such as the record ID column, attribute list, and default output CSV.
- `[[datasets]]` – dataset presets referencing your CSVs. Each entry selects comparison, blocking, filter, and classification profiles (or overrides specific parameters like ANN `k`).
- `[blocking.defaults]` – base partition attributes and ANN parameters applied unless a dataset overrides them.
- `[comparison]` – named profiles that map attributes to similarity functions (GPU or CPU implementations from `comparison.py`).
- `[filters]` – reusable high-precision filter profiles. Each profile defines mandatory conditions (`enforce_all`) and a set of rule groups that are OR-combined.
- `[classification]` – classifier defaults (`n_estimators`, `base_threshold`) and profiles that set precision/recall targets and threshold offsets.

### `src/config.py`
- Logging configuration (`LOG_LEVEL`, `LOG_FORMAT`).
- `Q_GRAM_LENGTH` for q-gram similarity functions (Jaccard/Dice).
- `USE_GPU_COMPARISON` default; the CLI overrides this when `--use-gpu` or `--no-gpu` is supplied.

### Advanced tuning in code (optional)
- `src/blocking.py` – extend or swap blocking strategies if you need custom logic beyond the configurable ANN/simple blocking.
- `src/comparison.py` – add new similarity functions and reference them in `comparison.profiles` within the TOML.
- `src/classification.py` – modify the ratio/max-depth grids or replace the classifier if you need a different learner.
- `src/evaluation.py` – adjust or add evaluation metrics.

## Adding new datasets
1. Place your CSVs somewhere accessible (inside or outside the repo).
2. Duplicate one of the `[[datasets]]` blocks in `config/pipeline.toml` and update the `dataset_a`, `dataset_b`, and `truth` paths.
3. Point the new dataset at existing profiles (`comparison_profile`, `profile`, `classification_profile`) or create new ones in the corresponding sections.
4. Adjust overrides such as `blocking_ann_attributes`, `blocking_k_neighbors`, or `output_csv` if the default behaviour does not suit your data.
5. Run `python src/recordLinkage.py --dataset <your-key>` to validate the configuration. Use `--skip-filters` while exploring new similarity thresholds.

## Troubleshooting
- **CUDA out of memory** – reduce ANN `K_NEIGHBORS`, increase filtering thresholds, or disable GPU comparisons to fall back on CPU.
- **Missing RAPIDS libraries** – rebuild the environment from `conda_env.txt` or install the specific packages (`conda install -n rapids-rl -c rapidsai -c conda-forge cudf cuml cupy faiss-gpu`).
- **Slow CPU runs** – set `USE_GPU_COMPARISON=True` and verify the environment is activated on a GPU-enabled host.

## Example workflow
```bash
conda activate rapids-rl
python src/recordLinkage.py --dataset clean_100000
```
Adjust `LOG_LEVEL` in `src/config.py` if you need more verbose logs during the run, or point to a different config file with `--config`. After completion, inspect the output CSV defined in your dataset entry and the logged precision/recall metrics to validate linkage quality. Iterate on the TOML profiles to meet your data quality and runtime targets.

## Further Reading

### Library documentation
- RAPIDS cuDF documentation: https://docs.rapids.ai/api/cudf/stable/
- RAPIDS cuML documentation (RandomForest, TF-IDF, ANN utilities): https://docs.rapids.ai/api/cuml/stable/
- CuPy user guide: https://docs.cupy.dev/en/stable/
- Numba CUDA programming guide: https://numba.readthedocs.io/en/stable/cuda/index.html
- FAISS GPU reference: https://faiss.ai/
- RapidFuzz API reference: https://rapidfuzz.github.io/rapidfuzz/

### GPU-accelerated record linkage & entity resolution
- Y. Jiang, P. Christen, U. Rahm, “Scaling blocking for record linkage on multi-core and GPU processors,” *The VLDB Journal* 26, 811–835 (2017). https://doi.org/10.1007/s00778-016-0440-4
- NVIDIA Developer Blog, “Accelerating Entity Resolution with RAPIDS,” highlights GPU-accelerated record linkage workflows using cuDF and cuML. https://developer.nvidia.com/blog/accelerating-entity-resolution-with-rapids/
- RAPIDS notebooks-contrib, “Record Linkage with RAPIDS cuDF and cuML,” end-to-end GPU pipeline example. https://github.com/rapidsai/notebooks-contrib/blob/main/intermediate_notebooks/rapids_record_linkage.ipynb

### Custom GPU kernels with Numba
- NVIDIA Developer Blog, “CUDA Python: Using Numba to Accelerate Python,” walkthrough of writing and optimizing custom CUDA kernels in Python. https://developer.nvidia.com/blog/cuda-python-numba/
- S. K. Lam, A. Pitrou, S. Seibert, “Numba: A LLVM-based Python JIT compiler,” *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*, 2015. https://doi.org/10.1145/2833157.2833162
- RAPIDS documentation on user-defined functions with Numba (applies to custom kernels executed inside cuDF): https://docs.rapids.ai/api/cudf/stable/udf/intro/
