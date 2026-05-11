"""Dataset loaders for dimensionality reduction experiments."""

from .load_data import (
    ALL_DATASET_KEYS,
    DatasetBundle,
    load_20newsgroups_tfidf,
    load_coil20,
    load_dataset,
    load_iris,
    load_mnist,
    load_pbmc3k,
    load_swiss_roll,
    load_wine,
    make_circles_,
)

__all__ = [
    "ALL_DATASET_KEYS",
    "DatasetBundle",
    "load_20newsgroups_tfidf",
    "load_coil20",
    "load_dataset",
    "load_iris",
    "load_mnist",
    "load_pbmc3k",
    "load_swiss_roll",
    "load_wine",
    "make_circles_",
]
