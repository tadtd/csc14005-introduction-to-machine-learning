# Lab 2 - Dimensionality Reduction

This repository contains the materials for CSC14005 - Introduction to Machine Learning, Lab 2. The topic is Chapter 12, *Dimensionality Reduction*, from *Foundations of Machine Learning* by Mohri, Rostamizadeh, and Talwalkar (2nd edition).

The project covers PCA, KPCA, Isomap, Laplacian Eigenmaps, LLE, and the Johnson--Lindenstrauss lemma. It also includes t-SNE and UMAP baselines, a Neg-t-SNE notebook, and a research discussion of PaCMAP.

| Student ID | Name |
|------------|------|
| 23120119 | Đỗ Tiến Đạt |
| 23120163 | Lê Nhật Minh Tâm |
| 23120137 | Triệu Tuấn Kiệt |
| 23120134 | Nguyễn Đăng Khoa |
| 23120105 | Huỳnh Mạnh Tường |

## Requirements

Python 3.13 or newer.

## How to run

Choose one of the two setup methods below.

### Option 1: Using `uv` (recommended)

From the project root:

```powershell
uv sync
```

To install optional extras when needed:

```powershell
uv sync --extra data
uv sync --extra contrastive
```

The `data` extra installs Scanpy and Pillow for the extended datasets. The `contrastive` extra installs `torch` and `contrastive-ne` for `code/experiments/contrastive_learning.ipynb`.

### Option 2: Using `pip`

Install all dependencies from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

This file includes the dependencies needed for the notebooks and the report scripts.

## Data

- MNIST should be placed at `code/data/raw/mnist.hdf5`, or you can use the automatic loader documented in [`code/data/README.md`](code/data/README.md).
- To download the supported datasets in advance, run:

```powershell
uv run python code/data/load_data.py
```

If you use `pip`, run the same script with your active Python environment instead:

```powershell
python code/data/load_data.py
```

## Reproducing the results

Generate the figures and metrics used in the main report:

```powershell
uv run python code/experiments/report_pca_kpca_isomap.py
```

If you are using `pip`, run:

```powershell
python code/experiments/report_pca_kpca_isomap.py
```

Build the LaTeX report with the provided scripts:

```powershell
.\scripts\compile_latex.ps1
```

To clean LaTeX artifacts before rebuilding:

```powershell
.\scripts\clean_artifacts.ps1
```

To clean and rebuild in one step:

```powershell
.\scripts\recompile.ps1
```

## Project files

| File | Description |
|------|-------------|
| [`code/experiments/report_pca_kpca_isomap.py`](code/experiments/report_pca_kpca_isomap.py) | Generates the figures and tables for the main experiment section with `SEED = 42`. |
| [`code/experiments/baseline.ipynb`](code/experiments/baseline.ipynb) | Baseline PCA, KPCA, Isomap, LLE, Laplacian Eigenmaps, t-SNE, and UMAP experiments on 10,000 MNIST samples with `SEED = 42`. |
| [`code/experiments/contrastive_learning.ipynb`](code/experiments/contrastive_learning.ipynb) | Neg-t-SNE (`cne.CNE`) extension and t-SNE/UMAP wrappers. |

## Structure

```text
lab-2/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- report/
|   |-- main.tex
|   |-- ref/
|   |   |-- ref.tex
|   |   `-- ref.bib
|   |-- main.aux
|   |-- main.bbl
|   |-- main.blg
|   |-- main.fdb_latexmk
|   |-- main.fls
|   |-- main.lof
|   |-- main.lot
|   |-- main.toc
|   |-- content/
|   `-- figures/
`-- code/
    |-- README.md
    |-- models/
    |-- experiments/
    |-- data/
    `-- utils/
```

See [`code/README.md`](code/README.md) for details about each source file and the from-scratch/wrapper scope.
