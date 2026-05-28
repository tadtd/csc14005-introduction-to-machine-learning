# Data Helpers (`load_data.py`)

This folder contains the shared dataset loaders and cache for the dimensionality reduction experiments. Large or downloaded data should live under `code/data/raw/`.

## Course dataset (Google Drive)

Manual copies of large files are hosted on Google Drive:

**[ML-LAB-2 — Google Drive](https://drive.google.com/drive/u/1/folders/1EacZk-p7jFkkpE4WnyKv4xTdL8j3wyMf)**

| Item on Drive | Place under `code/data/raw/` |
|---------------|------------------------------|
| `mnist.hdf5` | `code/data/raw/mnist.hdf5` |
| `coil-20/` (folder) | `code/data/raw/coil-20/` |

You do not have to download these if you are fine with automatic fetching (OpenML for MNIST, Columbia URL for COIL-20). The Drive folder is useful when downloads are slow or blocked.

`load_data.py` currently manages these datasets:

| Kind | Name(s) | Notes |
|------|---------|-------|
| Synthetic | `circles`, `swiss_roll` | Generated locally with scikit-learn. |
| Images | `mnist` | Auto-downloads from OpenML on first use and caches under `raw/.cache/openml/`. |
| Images | `coil20`, `coil_20`, `coil-20` | Auto-downloads the COIL-20 processed archive on first use and caches under `raw/.cache/coil20/`. |

Wine and Iris are used directly in the experiment scripts through scikit-learn, so they do not need to be prefetched into this folder.

## Use

```python
from code.data import load_dataset, DatasetBundle
```

Example:

```python
bundle = load_dataset("mnist")
X, y = bundle.X, bundle.y
```

## Prefetch

To warm the caches before running experiments:

```powershell
uv run python code/data/load_data.py
```

If you are using `pip`, run the same script with your active environment:

```powershell
python code/data/load_data.py
```

## Folder structure

```text
code/data/
├── __init__.py
├── load_data.py
├── README.md
└── raw/
    ├── mnist.hdf5
    ├── coil-20/
    └── .cache/
        ├── openml/
        └── coil20/
```

