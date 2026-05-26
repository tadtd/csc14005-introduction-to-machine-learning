from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from umap import UMAP as UmapLearn

from .base import BaseDR


class UMAP(BaseDR):
  """Thin wrapper around umap-learn for experiment notebooks."""

  def __init__(
    self,
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    n_jobs: int = 1,
    **kwargs: Any,
  ) -> None:
    kwargs.setdefault("center", False)
    super().__init__(n_components=n_components, **kwargs)
    self.n_neighbors = n_neighbors
    self.min_dist = min_dist
    self.metric = metric
    self.n_jobs = n_jobs
    self.embedding_: Optional[np.ndarray] = None
    self._X_fit: Optional[np.ndarray] = None

  def _fit(self, X: np.ndarray) -> None:
    self.embedding_ = self._run_umap(X)
    self._X_fit = X.copy()

  def _fit_transform(self, X: np.ndarray) -> np.ndarray:
    self._fit(X)
    if self.embedding_ is None:
      raise RuntimeError("UMAP failed to produce an embedding.")
    return self.embedding_

  def _transform(self, X: np.ndarray) -> np.ndarray:
    if self.embedding_ is None or self._X_fit is None:
      raise RuntimeError("UMAP has not been fitted.")
    if X.shape == self._X_fit.shape and np.allclose(X, self._X_fit):
      return self.embedding_
    raise NotImplementedError("umap-learn UMAP does not support out-of-sample transform.")

  def _run_umap(self, X: np.ndarray) -> np.ndarray:
    n_neighbors = min(self.n_neighbors, max(2, X.shape[0] - 1))
    model = UmapLearn(
      n_components=self.n_components,
      n_neighbors=n_neighbors,
      min_dist=self.min_dist,
      metric=self.metric,
      random_state=self.random_state,
      n_jobs=self.n_jobs,
    )
    return model.fit_transform(X)

  def get_params(self) -> Dict[str, Any]:
    return {
      **super().get_params(),
      "n_neighbors": self.n_neighbors,
      "min_dist": self.min_dist,
      "metric": self.metric,
      "n_jobs": self.n_jobs,
    }
