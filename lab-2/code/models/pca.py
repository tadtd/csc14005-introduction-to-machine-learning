from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

from .base import BaseDR


class PCA(BaseDR):
  """Principal Component Analysis implemented with NumPy SVD."""

  def __init__(
    self,
    n_components: int = 2,
    **kwargs: Any
  ) -> None:
    super().__init__(n_components=n_components, **kwargs)
    self.components_: Optional[np.ndarray] = None
    self.singular_values_: Optional[np.ndarray] = None
    self.explained_variance_: Optional[np.ndarray] = None
    self.explained_variance_ratio_: Optional[np.ndarray] = None

  def _fit(self, X: np.ndarray) -> None:
    max_components = min(X.shape)
    if self.n_components > max_components:
      raise ValueError(
        f"n_components ({self.n_components}) must be <= min(n_samples, n_features) "
        f"({max_components})."
      )

    _, singular_values, components = np.linalg.svd(X, full_matrices=False)
    denominator = max(X.shape[0] - 1, 1)
    all_variance = (singular_values ** 2) / denominator
    total_variance = float(np.sum(all_variance))

    self.components_ = components[: self.n_components]
    self.singular_values_ = singular_values[: self.n_components]
    self.explained_variance_ = all_variance[: self.n_components]
    if total_variance > 0.0:
      self.explained_variance_ratio_ = self.explained_variance_ / total_variance
    else:
      self.explained_variance_ratio_ = np.zeros(self.n_components, dtype=float)

  def _transform(self, X: np.ndarray) -> np.ndarray:
    if self.components_ is None:
      raise RuntimeError("PCA has not been fitted.")
    return X @ self.components_.T

  def _inverse_transform(self, Y: np.ndarray) -> np.ndarray:
    if self.components_ is None:
      raise RuntimeError("PCA has not been fitted.")
    return Y @ self.components_

  def reconstruction_error(self, X: np.ndarray) -> float:
    """Return mean squared reconstruction error per sample."""
    X = np.asarray(X, dtype=float)
    reconstructed = self.inverse_transform(self.transform(X))
    return float(np.mean(np.sum((X - reconstructed) ** 2, axis=1)))

  def get_params(self) -> Dict[str, Any]:
    return {**super().get_params()}