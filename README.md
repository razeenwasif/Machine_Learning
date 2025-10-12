# Machine Learning

Comprehensive collection of notebooks, scripts, and miniature projects covering core and advanced machine learning topics. The repository is organized by concept so you can jump straight to the material you need, whether you are revising foundational algorithms or exploring modern deep learning workflows.

## Repository Layout

| Path | Highlights |
| --- | --- |
| `Introduction_to_ML/` | Orientation notebooks that walk through course logistics and the broader ML pipeline. |
| `Linear-Regression/` | Gradient descent, PCA, and regression labs with supporting datasets under `data/`. |
| `Classification/`, `Clustering/`, `Decision-Trees/` | Supervised and unsupervised learning labs, mostly in notebook form. |
| `Bayesian_Logistic_Regression/` | Python scripts for Bayesian logistic regression; duplicated with a space in the folder name for compatibility with legacy paths. |
| `Kernel_regression/`, `Laplace_approximation/` | Kernel and Bayesian inference notebooks, including intermediate results and datasets. |
| `Neural-Network/`, `Transformer/` | Deep learning experiments ranging from classic MLPs to transformer-based models. |
| `Stable-Diffusion/`, `SIN-GAN/` | Generative modelling explorations and notes. |
| `Document_Analysis/`, `Visualisation/` | Notebook-based workflows for feature engineering, NLP, and visualization utilities. |
| `Projects/character_recognition`, `Projects/language-quiz` | Self-contained applied projects; the latter includes a `requirements.txt` for reproducible installs. |

Some topics appear twice (with and without underscores). These are intentional duplicates kept to preserve original course folder structures that may be referenced in notebooks.

## Getting Started

1. Clone the repository and create an isolated environment:
   ```bash
   git clone https://github.com/<your-username>/Machine_Learning.git
   cd Machine_Learning
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install the core notebook tooling and scientific stack:
   ```bash
   pip install --upgrade pip
   pip install jupyterlab numpy scipy pandas scikit-learn matplotlib seaborn
   ```
3. Install any project-specific requirements when needed (for example, `pip install -r Projects/language-quiz/requirements.txt`).

## Working with the Notebooks

- Launch Jupyter with `jupyter lab` (or `jupyter notebook`) from the repository root.
- Open the notebook of interest and execute cells sequentially; most notebooks assume data paths relative to their own directory.
- Check accompanying `.html` exports to preview results without executing the notebooks.

## Data and Results

- Datasets used in the labs are stored beside their notebooks (for example, `Linear-Regression/data/` and `Kernel_regression/03-dataset.csv`).
- Intermediate outputs (such as `.html` summary files) capture expected results, making it easy to compare your run with an instructor solution.

## Contributing and Maintenance

- Keep large datasets or model checkpoints outside the repository when possible; instead, document where to download them.
- When adding new material, follow the existing directory naming convention (`Topic-Name/` for notebooks, `Projects/<project-name>/` for larger efforts) and include brief README notes for context.
- Pull requests and improvements to documentation, reproducibility, or tests are welcome.
