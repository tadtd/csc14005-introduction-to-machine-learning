from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class BasisTransformer:
    """Minimal fit/transform interface for custom basis expansions."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BasisTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class PolynomialBasis(BasisTransformer):
    """Polynomial basis via scikit-learn's feature generator."""

    def __init__(
        self,
        degree: int = 2,
        *,
        include_bias: bool = False,
        interaction_only: bool = False,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self._poly = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PolynomialBasis":
        self._poly.fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._poly.transform(np.asarray(X, dtype=float))

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        return self._poly.get_feature_names_out(input_features)


class GaussianRBFBasis(BasisTransformer):
    """Gaussian radial basis expansion using training-set centers."""

    def __init__(
        self,
        n_centers: int = 10,
        *,
        gamma: float | None = None,
        length_scale: float | None = None,
        include_original_features: bool = False,
        center_selection: str = "random",
        random_state: int | None = 42,
    ) -> None:
        self.n_centers = n_centers
        self.gamma = gamma
        self.length_scale = length_scale
        self.include_original_features = include_original_features
        self.center_selection = center_selection
        self.random_state = random_state
        self.centers_: np.ndarray | None = None

    def _resolve_gamma(self) -> float:
        if self.gamma is not None:
            if self.gamma <= 0:
                raise ValueError("gamma must be positive.")
            return float(self.gamma)
        if self.length_scale is None or self.length_scale <= 0:
            raise ValueError("Provide either a positive gamma or length_scale.")
        return float(1.0 / (2.0 * self.length_scale * self.length_scale))

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GaussianRBFBasis":
        X = np.asarray(X, dtype=float)
        if self.n_centers <= 0:
            raise ValueError("n_centers must be positive.")

        n_samples = X.shape[0]
        n_centers = min(self.n_centers, n_samples)
        if self.center_selection == "first":
            indices = np.arange(n_centers)
        else:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=n_centers, replace=False)

        self.centers_ = X[indices].copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("Call fit before transform.")
        X = np.asarray(X, dtype=float)
        gamma = self._resolve_gamma()
        sq_dist = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(self.centers_**2, axis=1)
            - 2.0 * X @ self.centers_.T
        )
        sq_dist = np.maximum(sq_dist, 0.0)
        basis = np.exp(-gamma * sq_dist)
        if self.include_original_features:
            return np.hstack([X, basis])
        return basis

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("Call fit before get_feature_names_out.")
        names = [f"rbf_{idx}" for idx in range(self.centers_.shape[0])]
        if self.include_original_features:
            if input_features is None:
                input_features = [f"x{i}" for i in range(self.centers_.shape[1])]
            names = list(input_features) + names
        return np.asarray(names, dtype=object)


class SigmoidBasis(BasisTransformer):
    """Sigmoid basis functions with trainable centers and shared slope."""

    def __init__(
        self,
        n_centers: int = 10,
        *,
        gamma: float = 1.0,
        offset: float = 0.0,
        include_original_features: bool = False,
        random_state: int | None = 42,
    ) -> None:
        self.n_centers = n_centers
        self.gamma = gamma
        self.offset = offset
        self.include_original_features = include_original_features
        self.random_state = random_state
        self.centers_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SigmoidBasis":
        X = np.asarray(X, dtype=float)
        if self.n_centers <= 0:
            raise ValueError("n_centers must be positive.")
        n_centers = min(self.n_centers, X.shape[0])
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0], size=n_centers, replace=False)
        self.centers_ = X[indices].copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("Call fit before transform.")
        X = np.asarray(X, dtype=float)
        projection = self.gamma * (X @ self.centers_.T) + self.offset
        basis = 1.0 / (1.0 + np.exp(-np.clip(projection, -60.0, 60.0)))
        if self.include_original_features:
            return np.hstack([X, basis])
        return basis

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("Call fit before get_feature_names_out.")
        names = [f"sigmoid_{idx}" for idx in range(self.centers_.shape[0])]
        if self.include_original_features:
            if input_features is None:
                input_features = [f"x{i}" for i in range(self.centers_.shape[1])]
            names = list(input_features) + names
        return np.asarray(names, dtype=object)


class InteractionBasis(PolynomialBasis):
    """Convenience transformer for pairwise interaction terms."""

    def __init__(self, degree: int = 2, *, include_bias: bool = False) -> None:
        super().__init__(
            degree=degree,
            include_bias=include_bias,
            interaction_only=True,
        )
