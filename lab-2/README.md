# Lab 2 - Dimensionality Reduction

This repository contains Lab 2 for `CSC14005 - Introduction to Machine Learning`.
It focuses on dimensionality reduction theory, implementation, empirical evaluation, and report writing.

## Team

| Student ID | Name |
|------------|------|
| 23120119 | Do Tien Dat |
| 23120163 | Le Nhat Minh Tam |
| 23120137 | Trieu Tuan Kiet |
| 23120134 | Nguyen Dang Khoa |
| 23120105 | Huynh Manh Tuong |

## What is implemented

### From-scratch reducers (`code/models/`)

- PCA
- KPCA
- Isomap
- LLE
- Laplacian Eigenmaps

### Wrapper-based reducers (extended experiments)

- t-SNE (`sklearn.manifold.TSNE`)
- UMAP (`umap-learn`)
- Neg-t-SNE (`contrastive-ne`, class `cne.CNE`)

### Experiment and report assets

- Main reproducible script: `code/experiments/report_pca_kpca_isomap.py`
- Baseline notebook: `code/experiments/baseline.ipynb`
- Contrastive notebook: `code/experiments/contrastive_learning.ipynb`
- Notebook helper module (metrics/plots/workflow): `code/experiments/utils.py`
- Report source: `report/main.tex` with section files in `report/content/`

## Requirements

- Python `>= 3.13`
- `uv` (recommended) or `pip`
- LaTeX toolchain to build the report:
  - preferred: `latexmk`
  - fallback: `pdflatex` + `bibtex`

Build-tool note for contrastive dependencies:

- **Windows:** install Microsoft C++ Build Tools (MSVC)
- **Linux:** install compiler + Python headers (for example `build-essential`, `python3-dev`)

## Setup

Run from the `lab-2/` directory.

### Option 1 (recommended): `uv`

```powershell
uv sync
```

Optional extras:

```powershell
uv sync --extra data
uv sync --extra contrastive
```

- `data`: additional dataset-related dependencies (`scanpy`, `pillow`)
- `contrastive`: installs `torch` and `contrastive-ne`

### Option 2: `pip`

```powershell
python -m pip install -r requirements.txt
```

## Data

Shared dataset loading is implemented in `code/data/load_data.py`.

Supported loaders:

- Synthetic: `circles`, `swiss_roll`
- Image datasets: `mnist`, `coil20` (alias support included)

Data/cache location:

- `code/data/raw/`

Prefetch caches before running experiments:

```powershell
uv run python code/data/load_data.py
```

or:

```powershell
python code/data/load_data.py
```

See `code/data/README.md` for details.

## Reproduce the report figures

Run the main script:

```powershell
uv run python code/experiments/report_pca_kpca_isomap.py
```

or:

```powershell
python code/experiments/report_pca_kpca_isomap.py
```

This script generates the figures used for the PCA/KPCA/Isomap report section
and saves outputs to `report/figures/`.

## Run notebook experiments

### Baseline notebook

`code/experiments/baseline.ipynb` compares:

- PCA, KPCA, Isomap, LLE, Laplacian Eigenmaps
- t-SNE and UMAP baselines

### Contrastive notebook

`code/experiments/contrastive_learning.ipynb` explores Neg-t-SNE and related
comparisons against t-SNE/UMAP.

Notebook tips:

- Open notebooks from repository root (`lab-2/`)
- Select the Python kernel from `.venv`
- Ensure `uv sync` has completed before running all cells

## Build the LaTeX report

Compile:

```powershell
.\scripts\compile_latex.ps1
```

Clean artifacts:

```powershell
.\scripts\clean_artifacts.ps1
```

Clean + compile:

```powershell
.\scripts\recompile.ps1
```

`compile_latex.ps1` automatically prefers `latexmk`; if unavailable, it falls
back to `pdflatex`/`bibtex` passes.

Main files:

- Source: `report/main.tex`
- Output PDF: `report/main.pdf`

## Quick workflow (recommended)

```powershell
uv sync
uv run python code/data/load_data.py
uv run python code/experiments/report_pca_kpca_isomap.py
.\scripts\compile_latex.ps1
```

## Repository structure

```
lab-2/
├── README.md
├── pyproject.toml
├── requirements.txt
├── code/
│   ├── models/          # DR algorithms (PCA, KPCA, Isomap, LLE, Laplacian Eigenmaps, t-SNE, UMAP, Neg-t-SNE, …)
│   ├── experiments/     # report script, notebooks, experiments.utils (plots, metrics)
│   ├── utils/           # knn_graph.py (symmetric k-NN graph for CNE)
│   └── data/            # load_data.py, raw/
├── report/
│   ├── main.tex         # LaTeX report source
│   ├── content/         # section .tex files
│   ├── appendix/
│   ├── figures/         # generated plots for the report
│   └── ref/             # ref.bib, ref.tex
└── scripts/
    ├── clean_artifacts.ps1
    ├── compile_latex.ps1
    └── recompile.ps1
```

## Troubleshooting

| Error | Fix |
|------|-----|
| `Microsoft Visual C++ 14.0 or greater is required` | Install MSVC Build Tools, restart terminal, then run `uv sync` again. |
| `No module named 'cne'` | Install contrastive dependencies: `uv sync --extra contrastive`. |
| `mnist.hdf5` not found | Use `code/data/load_data.py` to prefetch data, or place files under `code/data/raw/`. |
| `No module named 'experiments'` in notebooks | Open notebook from `lab-2` root and use the correct `.venv` kernel. |
| LaTeX compile fails | Ensure `latexmk` (or `pdflatex` + `bibtex`) is installed and available in PATH. |
