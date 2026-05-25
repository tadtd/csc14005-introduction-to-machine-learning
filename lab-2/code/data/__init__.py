"""Dataset loaders for dimensionality reduction experiments."""

from .load_data import (
    ALL_DATASET_KEYS,
    DatasetBundle,
    load_coil20,
    load_dataset,
    load_mnist,
    load_swiss_roll,
    make_circles_,
)

__all__ = [
    "ALL_DATASET_KEYS",
    "DatasetBundle",
    "load_coil20",
    "load_dataset",
    "load_mnist",
    "load_swiss_roll",
    "make_circles_",
]
