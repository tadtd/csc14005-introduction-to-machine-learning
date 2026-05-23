from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist

from .base import BaseDR


class Isomap(BaseDR):
    """Isomap from k-nearest-neighbor graph distances and classical MDS."""

    def __init__(
        self,
        n_neighbors: int = 10,
        n_components: int = 2,
        *,
        eigen_tol: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("center", False)
        super().__init__(n_components=n_components, **kwargs)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol

        self.embedding_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.geodesic_distances_: Optional[np.ndarray] = None
        self.kernel_: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None

    def _fit(self, X: np.ndarray) -> None:
        n_samples = X.shape[0]
        if self.n_neighbors >= n_samples:
            raise ValueError("n_neighbors must be smaller than n_samples.")

        distances = cdist(X, X, metric="euclidean")
        graph = self._build_neighbor_graph(distances)
        graph = self._connect_components(graph, distances)
        geodesic = shortest_path(graph, directed=False, unweighted=False)
        if not np.isfinite(geodesic).all():
            raise ValueError(
                "The Isomap neighbor graph is disconnected. Increase n_neighbors "
                "or use a smaller/cleaner sample."
            )

        K = self._double_center(geodesic ** 2)
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        positive = eigenvalues > self.eigen_tol
        if int(np.sum(positive)) < self.n_components:
            raise ValueError(
                f"Isomap found only {int(np.sum(positive))} positive eigenvalues; "
                f"need {self.n_components}."
            )

        self.eigenvalues_ = eigenvalues[positive][: self.n_components]
        self.eigenvectors_ = eigenvectors[:, positive][:, : self.n_components]
        self.embedding_ = self.eigenvectors_ * np.sqrt(self.eigenvalues_)
        self.geodesic_distances_ = geodesic
        self.kernel_ = K
        self._X_fit = X.copy()

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.embedding_ is None or self._X_fit is None:
            raise RuntimeError("Isomap has not been fitted.")
        if X.shape == self._X_fit.shape and np.allclose(X, self._X_fit):
            return self.embedding_
        raise NotImplementedError("Out-of-sample extension for Isomap is not implemented.")

    def _build_neighbor_graph(self, distances: np.ndarray) -> sparse.csr_matrix:
        n_samples = distances.shape[0]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for i in range(n_samples):
            neighbors = np.argsort(distances[i])[1 : self.n_neighbors + 1]
            for j in neighbors:
                rows.append(i)
                cols.append(int(j))
                data.append(float(distances[i, j]))

        graph = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        return graph.maximum(graph.T)

    @staticmethod
    def _connect_components(graph: sparse.csr_matrix, distances: np.ndarray) -> sparse.csr_matrix:
        n_components, labels = connected_components(graph, directed=False)
        if n_components <= 1:
            return graph

        graph = graph.tolil(copy=True)
        connected = {0}
        remaining = set(range(1, n_components))

        while remaining:
            best: tuple[float, int, int, int] | None = None
            connected_mask = np.isin(labels, list(connected))
            connected_idx = np.flatnonzero(connected_mask)
            for comp in remaining:
                comp_idx = np.flatnonzero(labels == comp)
                block = distances[np.ix_(connected_idx, comp_idx)]
                local = int(np.argmin(block))
                row, col = np.unravel_index(local, block.shape)
                d = float(block[row, col])
                if best is None or d < best[0]:
                    best = (d, int(connected_idx[row]), int(comp_idx[col]), comp)
            if best is None:
                break
            d, i, j, comp = best
            graph[i, j] = d
            graph[j, i] = d
            connected.add(comp)
            remaining.remove(comp)

        return graph.tocsr()

    @staticmethod
    def _double_center(D2: np.ndarray) -> np.ndarray:
        row_mean = D2.mean(axis=1, keepdims=True)
        col_mean = D2.mean(axis=0, keepdims=True)
        all_mean = D2.mean()
        return -0.5 * (D2 - row_mean - col_mean + all_mean)

    def get_params(self) -> Dict[str, Any]:
        return {
            **super().get_params(),
            "n_neighbors": self.n_neighbors,
            "eigen_tol": self.eigen_tol,
        }
