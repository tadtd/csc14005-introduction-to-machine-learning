# Lab data (`load_data.py`)

Everything heavy goes under `**code/data/raw/**`. MNIST is downloaded automatically from OpenML and cached under `**raw/.cache/openml/**`; COIL-20 is downloaded automatically from the Columbia archive and cached under `**raw/.cache/coil20/**`.

## Use

```python
from code.data import load_dataset, DatasetBundle
```

## By dataset

| Kind      | Name(s)                         | Notes                                                                 |
| --------- | -------------------------------- | --------------------------------------------------------------------- |
| Synthetic | `circles`, `swiss_roll`         | Generated locally with scikit-learn.                                  |
| Images    | `mnist`                         | Auto-downloads from OpenML on first use and caches under `raw/.cache/openml/`. |
| Images    | `coil20`, `coil_20`, `coil-20`  | Auto-downloads the official COIL-20 processed archive on first use and caches under `raw/.cache/coil20/`. |

## Prefetch

```bash
uv run python code/data/load_data.py
```

## Folder structure

```
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

