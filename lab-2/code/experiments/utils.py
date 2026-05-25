"""Plotting helpers for dimensionality-reduction experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_embedding_scatter"]


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
	"""Plot a reduced embedding as a scatter chart.

	Parameters
	----------
	embedding:
		The reduced data matrix. The function uses the two columns given by
		``dimensions``.
	labels:
		Optional labels used to color the points.
	ax:
		Existing matplotlib axes. If omitted, a new figure is created.
	title:
		Optional plot title.
	dimensions:
		A pair of column indices used for the x/y coordinates.
	figsize:
		Figure size used when a new figure is created.
	cmap, s, alpha, edgecolors:
		Styling parameters forwarded to ``matplotlib.axes.Axes.scatter``.
	legend_max_items:
		Maximum number of distinct labels to list individually in the legend.
	numeric_bins:
		Number of legend bins to use when numeric labels have too many distinct values.
	show:
		Call ``plt.show()`` when a new figure is created.
	"""

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
