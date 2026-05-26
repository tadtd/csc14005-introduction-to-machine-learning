"""Plotting helpers for dimensionality-reduction experiments."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_embedding_spectrum(
    embeddings: Mapping[float, np.ndarray],
    y: np.ndarray,
    z_bars: Optional[Sequence[float]] = None,
    *,
    suptitle: str = "Neg-t-SNE embedding spectrum on MNIST",
    point_size: float = 1.0,
    alpha: float = 0.85,
    ncols: int = 3,
    figsize_per_ax: tuple[float, float] = (4.2, 4.0),
) -> plt.Figure:
    """
    Grid of 2D scatter plots, one panel per Z_bar.

    Parameters
    ----------
    embeddings : mapping
        Z_bar -> (n_samples, 2) coordinates.
    y : array
        Class labels for coloring.
    z_bars : sequence, optional
        Order of panels; defaults to sorted keys of ``embeddings``.
    """
    if z_bars is None:
        z_bars = sorted(embeddings.keys())
    z_bars = list(z_bars)
    n_plots = len(z_bars)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, z in zip(axes_flat, z_bars):
        z_key = float(z)
        embd = embeddings[z_key]
        ax.scatter(
            embd[:, 0],
            embd[:, 1],
            c=y,
            s=point_size,
            alpha=alpha,
            cmap="tab10",
            edgecolors="none",
        )
        ax.set_title(r"$\bar{Z}$" + f" = {z_key:.0e}")
        ax.set_aspect("equal", "datalim")
        ax.axis("off")

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=14)
    return fig
