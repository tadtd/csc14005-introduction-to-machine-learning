import numpy as np
from .base import BaseDR

class LaplacianEigenmaps(BaseDR):
    def __init__(self, n_neighbors=5, n_components=2, sigma=1.0, **kwargs):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        super().__init__(n_components=n_components, **kwargs)
        self.k = n_neighbors
        self.sigma = float(sigma)
        self.embedding_ = None
        self._X_fit = None

    def _fit(self, X: np.ndarray) -> None:
        distance_matrix = self._compute_distance_matrix(X)
        neighbors = self._find_neighbors(distance_matrix, self.k)

        W = self._build_weight_matrix(distance_matrix, neighbors, self.sigma)
        L = self._build_laplacian(W)

        _, eigenvectors = np.linalg.eigh(L)
        selected_indices = np.arange(1, self.n_components + 1)
        self.embedding_ = eigenvectors[:, selected_indices]
        self._X_fit = X

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if X is self._X_fit:
            return self.embedding_
        if (self._X_fit is not None
                and X.shape == self._X_fit.shape
                and np.allclose(X, self._X_fit)):
            return self.embedding_
        raise NotImplementedError("Out-of-sample extension for Laplacian Eigenmaps is not implemented.")

    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        sq = np.sum(X ** 2, axis=1)
        D2 = sq[:, None] + sq[None, :] - 2 * (X @ X.T)
        return np.sqrt(np.maximum(D2, 0))   # clip avoid negative sqrt due to numerical issues

    def _find_neighbors(self, distance_matrix: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(distance_matrix, axis=1)[:, 1:k + 1]

    def _build_weight_matrix(self, distance_matrix: np.ndarray,
                              neighbors: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        n = distance_matrix.shape[0]
        W = np.zeros((n, n))

        rows = np.repeat(np.arange(n), self.k)
        cols = neighbors.ravel()

        w_vals = np.exp(-(distance_matrix[rows, cols] ** 2) / (sigma ** 2))
        W[rows, cols] = w_vals
        W[cols, rows] = w_vals
        return W

    def _build_laplacian(self, W: np.ndarray) -> np.ndarray:
        L = -W.copy()
        np.fill_diagonal(L, W.sum(axis=1)) # fill the diagonal with the sum of the weights
        return L
