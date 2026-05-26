# Lab 2 — Dimensionality Reduction

Experiments on dimensionality reduction: PCA, KPCA, Isomap, LLE, Laplacian Eigenmaps, t-SNE, UMAP.

## Requirements

- **Python** ≥ 3.13
- **[uv](https://docs.astral.sh/uv/)** for environment and dependency management
- **Windows:** [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (MSVC) — required to build `annoy`, a dependency of `contrastive-ne` (see below)
- **Linux:** `build-essential` and Python headers, e.g. `sudo apt install build-essential python3-dev`

## Quick start

From the `lab-2` directory:

```powershell
cd lab-2
uv sync
```

Activate the virtual environment:

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Verify `contrastive-ne` (used in code as `from cne import CNE`):

```powershell
uv run python -c "from cne import CNE; print('cne OK')"
```

### Optional dependencies

For Scanpy / Pillow (extra datasets):

```powershell
uv sync --extra data
```

## Windows — MSVC (Build Tools)

The PyPI package [`contrastive-ne`](https://pypi.org/project/contrastive-ne/) depends on `annoy`, which must be compiled from source on Windows. If `uv sync` fails with *Microsoft Visual C++ 14.0 or greater is required*:

1. Install **Visual Studio Build Tools 2022** with the **Desktop development with C++** workload (or at least **MSVC** + **Windows SDK**).
2. Quick install via `winget` (PowerShell):

   ```powershell
   winget install -e --id Microsoft.VisualStudio.2022.BuildTools `
     --accept-package-agreements --accept-source-agreements `
     --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

   If `winget` is not on your PATH:

   ```powershell
   & "$env:LocalAppData\Microsoft\WindowsApps\winget.exe" install -e --id Microsoft.VisualStudio.2022.BuildTools `
     --accept-package-agreements --accept-source-agreements `
     --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

3. **Close and reopen your terminal**, then run:

   ```powershell
   uv sync
   ```

4. If `cl` is still not found, open **Developer PowerShell for VS 2022** from the Start menu, `cd` into `lab-2`, and run `uv sync` again.

Check that the compiler is available:

```powershell
where cl
```

> **Note:** MSVC is Windows-only. On Linux and macOS, use GCC or Clang (often preinstalled or available via your package manager).

## Data

- **MNIST** (baseline and contrastive notebooks): place `code/data/raw/mnist.hdf5`, or use the auto-download loader — see [`code/data/README.md`](code/data/README.md).
- Prefetch MNIST / COIL-20:

  ```powershell
  uv run python code/data/load_data.py
  ```

## Running experiments

| Notebook | Description |
|----------|-------------|
| [`code/experiments/baseline.ipynb`](code/experiments/baseline.ipynb) | Baselines: PCA, KPCA, Isomap, LLE, Laplacian Eigenmaps, t-SNE, UMAP |
| [`code/experiments/contrastive_learning.ipynb`](code/experiments/contrastive_learning.ipynb) | Neg-t-SNE spectrum (Damrich et al., ICLR 2023) vs sklearn t-SNE / UMAP |

In Jupyter or VS Code, select the **Python 3.13** kernel from `.venv` in `lab-2`. Notebooks prepend `lab_root/code` to `sys.path` and import helpers from `experiments.utils` (plotting, metrics, baselines). Use `lab-2` as the working directory when opening a terminal or kernel.

## Neg-t-SNE / contrastive-ne

- **Dependency:** [`contrastive-ne`](https://pypi.org/project/contrastive-ne/) from **PyPI** (no local `external/contrastive-ne` checkout).
- **Lab code:** [`code/models/neg_tsne.py`](code/models/neg_tsne.py) wraps `cne.CNE` with a fixed normalization constant $\bar{Z}$.
- **Spectrum:** sweep $\bar{Z}$ from **t-SNE-like** ($100 \cdot n$) to **UMAP-like** ($n^2/m$) via `geomspace`, with early exaggeration at $n^2/m$ (Damrich et al., 2023).

## Project layout

```
lab-2/
├── pyproject.toml
├── code/
│   ├── models/          # DR algorithms (PCA, t-SNE, UMAP, NegTSNE, …)
│   ├── experiments/     # Notebooks + experiments.utils (plots, metrics)
│   ├── utils/           # knn_graph.py (sklearn kNN for CNE)
│   └── data/            # load_data.py, raw/
└── README.md
```

## Troubleshooting

| Error | What to do |
|-------|------------|
| `Distribution not found at .../external/contrastive-ne` | Remove any old `[tool.uv.sources]` entry; run `uv lock` then `uv sync` (PyPI only). |
| `Microsoft Visual C++ 14.0 or greater is required` | Install MSVC Build Tools (Windows section above), restart the terminal, `uv sync`. |
| `No module named 'cne'` | Run `uv sync` in `lab-2` and select the `.venv` kernel. |
| `mnist.hdf5` not found | Copy the file to `code/data/raw/` or use `load_data` — see [`code/data/README.md`](code/data/README.md). |
| `No module named 'experiments'` | Run notebooks from `lab-2` and execute the path-setup cell so `code/` is on `sys.path`. |
