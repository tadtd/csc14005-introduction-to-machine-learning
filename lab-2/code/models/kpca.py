from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.spatial.distance import cdist

from .base import BaseDR


KernelName = Literal["linear", "rbf", "poly", "sigmoid", "precomputed"]


class KPCA(BaseDR):
    """Kernel PCA implemented from the centered Gram matrix."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        kernel: KernelName = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        eigen_tol: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("center", False)
        super().__init__(n_components=n_components, **kwargs)
        if kernel not in {"linear", "rbf", "poly", "sigmoid", "precomputed"}:
            raise ValueError(f"Unsupported kernel {kernel!r}.")
        if degree < 1:
            raise ValueError("degree must be >= 1.")
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.eigen_tol = eigen_tol

        self.X_fit_: Optional[np.ndarray] = None
        self.alphas_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.embedding_: Optional[np.ndarray] = None
        self.K_fit_rows_mean_: Optional[np.ndarray] = None
        self.K_fit_all_mean_: Optional[float] = None

    def _fit(self, X: np.ndarray) -> None:
        if self.kernel == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("For kernel='precomputed', X must be a square Gram matrix.")
            K = np.asarray(X, dtype=float)
            self.X_fit_ = None
        else:
            self.X_fit_ = X.copy()
            K = self._kernel(X, X)

        Kc = self._center_fit_kernel(K)
        eigenvalues, eigenvectors = np.linalg.eigh(Kc)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        positive = eigenvalues > self.eigen_tol
        if int(np.sum(positive)) < self.n_components:
            raise ValueError(
                f"KPCA found only {int(np.sum(positive))} positive eigenvalues; "
                f"need {self.n_components}."
            )
        eigenvalues = eigenvalues[positive][: self.n_components]
        eigenvectors = eigenvectors[:, positive][:, : self.n_components]

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.alphas_ = eigenvectors / np.sqrt(eigenvalues)
        self.embedding_ = eigenvectors * np.sqrt(eigenvalues)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.alphas_ is None:
            raise RuntimeError("KPCA has not been fitted.")
        if self.kernel == "precomputed":
            K = np.asarray(X, dtype=float)
            if K.shape[1] != self._n_samples_fit:
                raise ValueError(
                    f"Precomputed kernel must have shape (n_samples, {self._n_samples_fit}), "
                    f"got {K.shape}."
                )
        else:
            if self.X_fit_ is None:
                raise RuntimeError("Missing fitted samples for KPCA transform.")
            K = self._kernel(X, self.X_fit_)
        return self._center_transform_kernel(K) @ self.alphas_

    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
        if self.kernel == "linear":
            return X @ Y.T
        if self.kernel == "rbf":
            return np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))
        if self.kernel == "poly":
            return (gamma * (X @ Y.T) + self.coef0) ** self.degree
        if self.kernel == "sigmoid":
            return np.tanh(gamma * (X @ Y.T) + self.coef0)
        raise ValueError("_kernel is not used for precomputed kernels.")

    def _center_fit_kernel(self, K: np.ndarray) -> np.ndarray:
        self.K_fit_rows_mean_ = K.mean(axis=0)
        self.K_fit_all_mean_ = float(K.mean())
        return K - K.mean(axis=1, keepdims=True) - self.K_fit_rows_mean_[None, :] + self.K_fit_all_mean_

    def _center_transform_kernel(self, K: np.ndarray) -> np.ndarray:
        if self.K_fit_rows_mean_ is None or self.K_fit_all_mean_ is None:
            raise RuntimeError("Missing fitted kernel centering statistics.")
        return K - K.mean(axis=1, keepdims=True) - self.K_fit_rows_mean_[None, :] + self.K_fit_all_mean_

    def get_params(self) -> Dict[str, Any]:
        return {
            **super().get_params(),
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "eigen_tol": self.eigen_tol,
        }
