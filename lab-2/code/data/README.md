# Lab data (`load_data.py`)

Everything heavy goes under `**code/data/raw/**`: sklearn/OpenML cache (`.cache/sklearn/`), scanpy cache (`.cache/scanpy/`), and scanpy’s `**datasetdir**` (so PBMC lands as `**raw/pbmc3k_raw.h5ad**`, not `./data/`).

## Use

```python
from code.data import load_dataset, DatasetBundle
```

## By dataset


| Kind           | Name(s)                      | Notes                                                                                                                                                                                                                         |
| -------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Synthetic      | `swiss_roll`, `circles`      | No download                                                                                                                                                                                                                   |
| Tabular        | `iris`, `wine`               | Bundled in sklearn                                                                                                                                                                                                            |
| Images         | `mnist`                      | Local `**raw/mnist.hdf5**` ([PyMVPA / LeCun MNIST](https://www.pymvpa.org/datadb/mnist.html), [data mirror](http://data.pymvpa.org/datasets/mnist/)); optional kwarg `hdf5_path`                                              |
| Images (local) | `coil20`, `coil-20`          | [COIL-20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) → `**raw/coil-20/**` or `**raw/coil-20-proc/**` (same under `code/data/` still works). Nested `obj1`…`obj20` or flat `obj10__0.png`. Env: `COIL20_*`. |
| Text           | `20newsgroups`, `newsgroups` | Fetch + TF-IDF                                                                                                                                                                                                                |
| scRNA          | `pbmc3k`                     | `scanpy.datasets.pbmc3k()`; optional override file `**raw/pbmc3k.h5ad**`                                                                                                                                                      |


## Prefetch

```bash
uv run python code/data/load_data.py
```

## Folder structure

```
code/data/
├── __init__.py                 # Python package marker
├── load_data.py               # Main data loading module
├── README.md                  # This file (documentation)
└── raw/                       # Raw data directory for datasets
    ├── mnist.hdf5             # MNIST dataset (PyMVPA HDF5 format)
    ├── pbmc3k_raw.h5ad        # PBMC 3K single-cell RNA data (AnnData format)
    ├── coil-20/               # COIL-20 object images
    │   ├── obj1/              # Images of object 1
    │   ├── obj2/              # Images of object 2
    │   └── ...
    │   └── obj20/             # Images of object 20
    └── .cache/                # Auto-generated cache directories
        ├── sklearn/           # sklearn datasets cache
        └── scanpy/            # scanpy datasets cache
```