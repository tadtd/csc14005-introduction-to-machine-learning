"""
Abstract base class for dimensionality reduction algorithms.
All concrete reducers (PCA, KPCA, Isomap, t-SNE, UMAP, …) inherit from this.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDR(ABC):
  """
  Abstract base class for dimensionality reduction.

  All subclasses must implement:
    _fit(X): learn internal state from X
    _transform(X): project X into the low-dim space
    _fit_transform(X): (optional override) learn & project in one pass
                            Default calls _fit then _transform.

  The public API (`fit`, `transform`, `fit_transform`) adds:
    - input validation
    - centering / scaling (optional)
    - fitted-state guards
    - timing & metadata
  """

  def __init__(
    self,
    n_components: int = 2,
    *,
    center: bool = True,
    random_state: Optional[int] = None,
  ) -> None:
    """
    Parameters:
    - n_components : int
      Dimensionality of the embedding space.
    - center : bool
      Subtract the column mean before fitting/transforming.
      Some algorithms (e.g. kernel methods) handle centering internally;
      subclasses may override this default.
    - random_state : int or None
      Seed for reproducible stochastic algorithms.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components!r}")

    self.n_components = n_components
    self.center = center
    self.random_state = random_state

    # Populated after fit()
    self._is_fitted: bool = False
    self._n_samples_fit: int = 0
    self._n_features_fit: int = 0
    self._mean_: Optional[np.ndarray] = None  # shape (n_features,)
    self._fit_time: float = 0.0
    self._transform_time: float = 0.0

    # Seeded RNG exposed to subclasses
    self._rng: np.random.Generator = np.random.default_rng(random_state)

  @abstractmethod
  def _fit(self, X: np.ndarray) -> None:
    """
    Learn algorithm-specific parameters from *pre-processed* X.
    Results must be stored on self.
    """
    raise NotImplementedError("Subclasses must implement _fit")

  @abstractmethod
  def _transform(self, X: np.ndarray) -> np.ndarray:
    """
    Project pre-processed X into the embedding space.

    Returns:
    Y: ndarray, shape (n_samples, n_components)
    """
    raise NotImplementedError("Subclasses must implement _transform")

  def _fit_transform(self, X: np.ndarray) -> np.ndarray:
    """
    Learn parameters and return the embedding for X in one pass.
    Override when a joint computation is cheaper (e.g. t-SNE, UMAP).

    Default: calls _fit then _transform.
    """
    self._fit(X)
    return self._transform(X)

  def _inverse_transform(self, Y: np.ndarray) -> np.ndarray: 
    """
    Map the embedding back to the original space (approximate).
    Override in subclasses that support reconstruction (e.g. PCA).
    """
    raise NotImplementedError(f"{type(self).__name__} does not implement inverse_transform.")

  def fit(self, X: np.ndarray) -> "BaseDR":
    """Fit the model to X."""
    X = self._validate(X)
    Xp = self._preprocess(X, fitting=True)

    t0 = time.perf_counter()
    self._fit(Xp)
    self._fit_time = time.perf_counter() - t0

    self._is_fitted = True
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    """
    Project X into the embedding space.

    Parameters
    X : array-like, shape (n_samples, n_features)

    Returns
    Y : ndarray, shape (n_samples, n_components)
    """
    self._check_fitted()
    X = self._validate(X, expected_features=self._n_features_fit)
    Xp = self._preprocess(X, fitting=False)

    t0 = time.perf_counter()
    Y = self._transform(Xp)
    self._transform_time = time.perf_counter() - t0

    return self._validate_output(Y)

  def fit_transform(self, X: np.ndarray) -> np.ndarray:
    """
    Fit the model to X and return its embedding.

    Parameters
    X : array-like, shape (n_samples, n_features)

    Returns
    Y : ndarray, shape (n_samples, n_components)
    """
    X = self._validate(X)
    Xp = self._preprocess(X, fitting=True)

    t0 = time.perf_counter()
    Y = self._fit_transform(Xp)
    self._fit_time = time.perf_counter() - t0

    self._is_fitted = True
    return self._validate_output(Y)

  def inverse_transform(self, Y: np.ndarray) -> np.ndarray: 
    """
    Reconstruct original-space approximation from embedding Y.

    Parameters:
    - Y : array-like, shape (n_samples, n_components)

    Returns:
    - X_approx : ndarray, shape (n_samples, n_features)
    """
    self._check_fitted()
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != self.n_components:
      raise ValueError(f"Y must have shape (n_samples, {self.n_components}), got {Y.shape}")
    X_approx = self._inverse_transform(Y)
    # Undo centering
    if self.center and self._mean_ is not None:
      X_approx = X_approx + self._mean_
    return X_approx

  def _validate(
    self,
    X: np.ndarray ,
    *,
    expected_features: Optional[int] = None,
  ) -> np.ndarray:
    """Convert to float64 ndarray and sanity-check shape."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
      raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
    n_samples, n_features = X.shape
    if n_samples < 1:
      raise ValueError("X must contain at least one sample.")
    if n_features < 1:
      raise ValueError("X must contain at least one feature.")
    if n_samples < self.n_components:
      raise ValueError(f"n_samples ({n_samples}) < n_components ({self.n_components}).")
    if expected_features is not None and n_features != expected_features:
      raise ValueError(f"X has {n_features} features; expected {expected_features} (from training data).")
    if not np.isfinite(X).all():
      raise ValueError("X contains NaN or Inf values.")
    return X

  def _preprocess(self, X: np.ndarray, *, fitting: bool) -> np.ndarray:
    """Center (and optionally scale) the data."""
    if self.center:
      if fitting:
        self._mean_ = X.mean(axis=0)
        self._n_samples_fit, self._n_features_fit = X.shape
      X = X - self._mean_
    elif fitting:
      self._n_samples_fit, self._n_features_fit = X.shape
    return X

  def _validate_output(self, Y: np.ndarray) -> np.ndarray:
    """Ensure the subclass returned a well-shaped embedding."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
      raise RuntimeError(f"_transform must return a 2-D array, got shape {Y.shape}")
    if Y.shape[1] != self.n_components:
      raise RuntimeError(f"_transform must return shape (n_samples, {self.n_components}), got {Y.shape}")
    if not np.isfinite(Y).all():
      raise RuntimeError("_transform returned NaN or Inf values.")
    return Y

  def _check_fitted(self) -> None:
      if not self._is_fitted:
        raise RuntimeError(f"This {type(self).__name__} instance is not fitted yet. Call fit() or fit_transform() first.")

  def __repr__(self) -> str:
    params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
    return f"{type(self).__name__}({params})"

  def get_params(self) -> dict:
    """Return constructor parameters (sklearn-compatible)."""
    return {
      "n_components": self.n_components,
      "center": self.center,
      "random_state": self.random_state,
    }

  @property
  def is_fitted(self) -> bool:
    return self._is_fitted

  @property
  def fit_time(self) -> float:
    return self._fit_time

  @property
  def transform_time(self) -> float:
    return self._transform_time
