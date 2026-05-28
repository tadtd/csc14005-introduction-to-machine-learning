# Lab 2 - Dimensionality Reduction

Lab 2 for **CSC14005 - Introduction to Machine Learning**: dimensionality
reduction theory, from-scratch implementations, baseline experiments, and a
LaTeX report.

## Download (full source + report)

The complete submission bundle (source code, report, scripts, and project
metadata) is available on Google Drive:

**[ml-lab-2-code — Google Drive](https://drive.google.com/drive/folders/1o9qE7YEfH3MTnR-EyIGOmkCo16rRXGG9)**

That folder mirrors the project layout below (`code/`, `report/`, `scripts/`,
`pyproject.toml`, `README.md`, etc.). Use it if you only need a one-click
download without cloning this repository.

Large experiment datasets (MNIST, COIL-20) are hosted separately — see
[`code/data/README.md`](code/data/README.md).

## Team

| Student ID | Name |
|------------|------|
| 23120119 | Do Tien Dat |
| 23120163 | Le Nhat Minh Tam |
| 23120137 | Trieu Tuan Kiet |
| 23120134 | Nguyen Dang Khoa |
| 23120105 | Huynh Manh Tuong |

## Scope

### From-scratch reducers (`code/models/`)

- PCA, KPCA, Isomap, LLE, Laplacian Eigenmaps

### Wrapper-based reducers (extended experiments)

- t-SNE (`sklearn.manifold.TSNE`)
- UMAP (`umap-learn`)
- Neg-t-SNE (`contrastive-ne`, `cne.CNE`)

### Experiments and report

| Asset | Role |
|-------|------|
| `code/experiments/report_pca_kpca_isomap.py` | Reproduces PCA / KPCA / Isomap figures for the report (`SEED = 42`) |
| `code/experiments/baseline.ipynb` | Full baseline on 4 datasets (PCA, KPCA, Isomap, LLE, Laplacian Eigenmaps, t-SNE, UMAP) |
| `code/experiments/contrastive_learning.ipynb` | Neg-t-SNE extension vs t-SNE / UMAP |
| `code/experiments/utils.py` | Plotting, metrics, baseline workflow helpers |
| `report/main.tex` | LaTeX report source → `report/main.pdf` |

More implementation notes: [`code/README.md`](code/README.md). Report build
details: [`report/README.md`](report/README.md).

## Requirements

- Python `>= 3.13`
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- LaTeX: `latexmk` preferred, or `pdflatex` + `bibtex`
- **Windows (Neg-t-SNE):** [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) — `contrastive-ne` depends on `annoy`, which is built from source
- **Linux:** `build-essential`, `python3-dev` (or equivalent)

## Setup

From the `lab-2/` directory:

```powershell
uv sync
```

Optional extras:

```powershell
uv sync --extra data        # scanpy, pillow (optional dataset tooling)
uv sync --extra contrastive # torch, contrastive-ne (Neg-t-SNE notebook)
```

Or with pip:

```powershell
python -m pip install -r requirements.txt
```

Activate the environment (optional):

```powershell
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
```

```bash
source .venv/bin/activate        # Linux / macOS
```

Verify Neg-t-SNE dependency:

```powershell
uv run python -c "from cne import CNE; print('cne OK')"
```

## Data

| Dataset | How it is loaded |
|---------|------------------|
| `circles`, `swiss_roll` | Generated via scikit-learn (`load_data.py`) |
| `mnist`, `coil20` | `load_data.py` — auto-download + cache under `code/data/raw/` |
| Wine, Iris | Built-in scikit-learn benchmarks (`load_wine`, `load_iris`) — not synthetic; used in theory / PCA checks, not in the 4-dataset baseline grid |

Prefetch MNIST / COIL-20 caches:

```powershell
uv run python code/data/load_data.py
```

Dataset files and the course Drive folder for manual download:
[`code/data/README.md`](code/data/README.md).

## Reproduce report figures

```powershell
uv run python code/experiments/report_pca_kpca_isomap.py
```

Outputs go to `report/figures/` (Wine PCA diagnostics, extended circles KPCA,
Swiss roll Isomap).

## Notebooks

Open from `lab-2/` and select the **Python 3.13** kernel from `.venv`.

| Notebook | Content |
|----------|---------|
| [`baseline.ipynb`](code/experiments/baseline.ipynb) | Baselines on `circles`, `swiss_roll`, `coil20`, `mnist` (10k MNIST subsample, fixed seed) |
| [`contrastive_learning.ipynb`](code/experiments/contrastive_learning.ipynb) | Neg-t-SNE spectrum vs t-SNE / UMAP — requires `uv sync --extra contrastive` |

## Build the report

```powershell
.\scripts\compile_latex.ps1      # build report/main.pdf
.\scripts\clean_artifacts.ps1    # remove LaTeX / Python cache files
.\scripts\recompile.ps1          # clean + build
```

## Quick workflow

```powershell
uv sync
uv run python code/data/load_data.py
uv run python code/experiments/report_pca_kpca_isomap.py
.\scripts\compile_latex.ps1
```

## Project layout

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
|-------|-----|
| `Microsoft Visual C++ 14.0 or greater is required` | Install MSVC Build Tools, restart the terminal, run `uv sync` again. |
| `No module named 'cne'` | `uv sync --extra contrastive`; use the `.venv` kernel in notebooks. |
| `mnist.hdf5` not found | Run `code/data/load_data.py` or copy from the dataset Drive — see [`code/data/README.md`](code/data/README.md). |
| `No module named 'experiments'` | Run notebooks from `lab-2/`; execute the path-setup cell so `code/` is on `sys.path`. |
| LaTeX compile fails | Install `latexmk` or `pdflatex` + `bibtex` and ensure they are on `PATH`. |
