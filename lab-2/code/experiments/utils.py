"""Plotting and experiment helpers for dimensionality-reduction notebooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "plot_embedding_scatter",
    "preprocess_for_dataset",
    "embedding_knn_accuracy",
    "evaluate_embedding",
    "run_dr",
    "collect_dr_result",
    "run_baseline_for_method",
    "plot_dataset_grid",
    "run_experiment_matrix",
    "results_to_dataframe",
    "plot_comparison_grid",
]

_IMAGE_DATASETS = frozenset({"coil20", "mnist", "coil_20", "coil-20"})


def preprocess_for_dataset(dataset_name: str, X: np.ndarray) -> np.ndarray:
    """Apply protocol preprocessing: scale image pixels to [0, 1]."""
    key = dataset_name.strip().lower().replace(" ", "_")
    X = np.asarray(X, dtype=float)
    if key in _IMAGE_DATASETS:
        return X / 255.0
    return X


def plot_embedding_scatter(
    embedding: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[Any] | None = None,
    *,
    ax: Any | None = None,
    title: str | None = None,
    dimensions: tuple[int, int] = (0, 1),
    figsize: tuple[float, float] = (13.5, 10.0),
    cmap: str = "tab20",
    s: float = 0.5,
    alpha: float = 0.8,
    edgecolors: str = "none",
    legend_max_items: int = 20,
    numeric_bins: int = 10,
    show: bool = True,
) -> Any:
    """Plot a reduced embedding as a scatter chart."""
    points = np.asarray(embedding)
    if points.ndim != 2:
        raise ValueError("embedding must be a 2D array-like structure")

    if len(dimensions) != 2:
        raise ValueError("dimensions must contain exactly two indices")

    x_dim, y_dim = dimensions
    max_dim = max(x_dim, y_dim)
    if max_dim >= points.shape[1]:
        raise ValueError(
            "embedding does not have enough columns for the requested dimensions"
        )

    coords = points[:, [x_dim, y_dim]]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if labels is None:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=s,
            alpha=alpha,
            edgecolors=edgecolors,
            linewidths=0,
        )
    else:
        label_values = np.asarray(labels)
        if label_values.shape[0] != coords.shape[0]:
            raise ValueError("labels must have the same length as embedding")

        if np.issubdtype(label_values.dtype, np.number):
            unique_labels, label_codes = np.unique(label_values, return_inverse=True)
            if unique_labels.size <= legend_max_items:
                palette = plt.get_cmap(cmap, unique_labels.size)
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=label_codes,
                    cmap=palette,
                    s=s,
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=0,
                )
            else:
                if numeric_bins < 1:
                    raise ValueError("numeric_bins must be at least 1")
                bin_edges = np.quantile(
                    label_values,
                    np.linspace(0.0, 1.0, numeric_bins + 1),
                )
                bin_edges = np.unique(bin_edges)
                if bin_edges.size < 2:
                    bin_edges = np.array([label_values.min(), label_values.max() + 1.0])
                bin_codes = np.digitize(label_values, bin_edges[1:-1], right=False)
                palette = plt.get_cmap(cmap, bin_edges.size - 1)
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=bin_codes,
                    cmap=palette,
                    s=s,
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=0,
                )
        else:
            unique_labels, label_codes = np.unique(
                label_values.astype(str), return_inverse=True
            )
            palette = plt.get_cmap(cmap, unique_labels.size)
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=label_codes,
                cmap=palette,
                s=s,
                alpha=alpha,
                edgecolors=edgecolors,
                linewidths=0,
            )

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(f"Component {x_dim + 1}")
    ax.set_ylabel(f"Component {y_dim + 1}")

    if show and ax.figure is not None:
        plt.show()

    return ax


def embedding_knn_accuracy(
    Y: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    """Majority-vote k-NN label accuracy in the embedding space."""
    y = np.asarray(y)
    k = min(n_neighbors + 1, len(Y))
    neighbors = (
        NearestNeighbors(n_neighbors=k)
        .fit(Y)
        .kneighbors(Y, return_distance=False)[:, 1:]
    )
    pred = []
    for row in neighbors:
        values, counts = np.unique(y[row], return_counts=True)
        pred.append(values[np.argmax(counts)])
    return float(np.mean(np.asarray(pred) == y))


def evaluate_embedding(
    X: np.ndarray,
    Y: np.ndarray,
    y: np.ndarray | None,
    *,
    trustworthiness_k: int = 12,
    knn_k: int = 5,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute standard DR evaluation metrics."""
    k_tw = min(trustworthiness_k, max(2, X.shape[0] - 2))
    metrics: dict[str, Any] = {
        f"trustworthiness@{trustworthiness_k}": round(
            float(trustworthiness(X, Y, n_neighbors=k_tw)),
            4,
        ),
    }
    if y is not None:
        metrics[f"knn_accuracy@{knn_k}"] = round(
            embedding_knn_accuracy(Y, y, n_neighbors=knn_k),
            4,
        )
    if extra:
        metrics.update(extra)
    return metrics


def run_dr(
    model: Any,
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    dataset: str,
    method: str,
    trustworthiness_k: int = 12,
    knn_k: int = 5,
    extra_metrics: Callable[[Any, np.ndarray], dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit-transform and evaluate one (dataset, method) pair."""
    Y = model.fit_transform(X)
    extra: dict[str, Any] | None = None
    if extra_metrics is not None:
        extra = extra_metrics(model, X)
    metrics = evaluate_embedding(
        X,
        Y,
        y,
        trustworthiness_k=trustworthiness_k,
        knn_k=knn_k,
        extra=extra,
    )
    metrics["fit_time_s"] = round(float(model.fit_time), 4)
    metrics["dataset"] = dataset
    metrics["method"] = method
    metrics["n_samples"] = int(X.shape[0])
    return Y, metrics


def collect_dr_result(
    Y: np.ndarray,
    metrics: dict[str, Any],
    *,
    store_embedding: bool = False,
) -> dict[str, Any]:
    """Build one results row; optionally attach the embedding array."""
    row = dict(metrics)
    if store_embedding:
        row["embedding"] = Y
    return row


def run_baseline_for_method(
    method_name: str,
    model_factory: Callable[[], Any],
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    dataset_order: Sequence[str],
    experiment_cfg: dict[str, Any],
    *,
    store_embeddings: bool = False,
    extra_metrics: Callable[[Any, np.ndarray], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Run baseline DR on each dataset in ``dataset_order``.

    Returns (result_rows, embeddings_by_dataset, labels_by_dataset).
    """
    tw_k = int(experiment_cfg.get("trustworthiness_k", 12))
    knn_k = int(experiment_cfg.get("knn_k", 5))
    rows: list[dict[str, Any]] = []
    embeddings: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}

    for name in dataset_order:
        if name not in datasets:
            raise KeyError(f"Dataset {name!r} not in datasets registry.")
        X_raw, y = datasets[name]
        X = preprocess_for_dataset(name, X_raw)
        model = model_factory()
        Y, metrics = run_dr(
            model,
            X,
            y,
            dataset=name,
            method=method_name,
            trustworthiness_k=tw_k,
            knn_k=knn_k,
            extra_metrics=extra_metrics,
        )
        rows.append(
            collect_dr_result(Y, metrics, store_embedding=store_embeddings)
        )
        embeddings[name] = Y
        labels[name] = np.asarray(y)

    return rows, embeddings, labels


def plot_dataset_grid(
    embeddings_by_name: Mapping[str, np.ndarray],
    labels_by_name: Mapping[str, np.ndarray],
    *,
    dataset_order: Sequence[str] | None = None,
    suptitle: str | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
) -> Any:
    """2x2 scatter grid for the four standard datasets."""
    order = list(dataset_order) if dataset_order is not None else list(embeddings_by_name.keys())
    if len(order) != 4:
        raise ValueError("plot_dataset_grid expects exactly 4 datasets.")

    kwargs = {"s": 8, "alpha": 0.7, "show": False}
    if scatter_kwargs:
        kwargs.update(scatter_kwargs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, name in zip(axes.ravel(), order):
        plot_embedding_scatter(
            embeddings_by_name[name],
            labels_by_name[name],
            title=name,
            ax=ax,
            **kwargs,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def run_experiment_matrix(
    method_factories: Mapping[str, Callable[[], Any]],
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    dataset_order: Sequence[str],
    experiment_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run baseline for every method in ``method_factories``."""
    all_rows: list[dict[str, Any]] = []
    for method_name, factory in method_factories.items():
        rows, _, _ = run_baseline_for_method(
            method_name,
            factory,
            datasets,
            dataset_order,
            experiment_cfg,
        )
        all_rows.extend(rows)
    return all_rows


def results_to_dataframe(rows: list[dict[str, Any]]) -> Any:
    """Convert result rows to a pandas DataFrame (drops embedding arrays)."""
    import pandas as pd

    clean = []
    for row in rows:
        clean.append({k: v for k, v in row.items() if k != "embedding"})
    return pd.DataFrame(clean)


def plot_comparison_grid(
    embeddings: Mapping[tuple[str, str], np.ndarray],
    labels: Mapping[str, np.ndarray],
    methods: Sequence[str],
    datasets: Sequence[str],
    *,
    scatter_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Grid with rows = datasets, columns = methods.

    ``embeddings`` keys are (dataset_name, method_name).
    """
    kwargs = {"s": 4, "alpha": 0.6, "show": False}
    if scatter_kwargs:
        kwargs.update(scatter_kwargs)

    n_rows, n_cols = len(datasets), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.0 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, ds in enumerate(datasets):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            key = (ds, method)
            if key not in embeddings:
                ax.set_visible(False)
                continue
            plot_embedding_scatter(
                embeddings[key],
                labels[ds],
                title=f"{method} / {ds}",
                ax=ax,
                **kwargs,
            )
    fig.tight_layout()
    return fig
