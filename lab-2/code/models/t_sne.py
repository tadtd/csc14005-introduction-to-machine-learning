from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.manifold import TSNE as SklearnTSNE

from .base import BaseDR


class TSNE(BaseDR):
    """Thin wrapper around sklearn.manifold.TSNE for the project bonus."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
        learning_rate: str | float = "auto",
        max_iter: int = 1000,
        init: str = "pca",
        metric: str = "euclidean",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("center", False)
        super().__init__(n_components=n_components, **kwargs)
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init = init
        self.metric = metric
        self.embedding_: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None

    def _fit(self, X: np.ndarray) -> None:
        self.embedding_ = self._run_tsne(X)
        self._X_fit = X.copy()

    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        self._fit(X)
        if self.embedding_ is None:
            raise RuntimeError("TSNE failed to produce an embedding.")
        return self.embedding_

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.embedding_ is None or self._X_fit is None:
            raise RuntimeError("TSNE has not been fitted.")
        if X.shape == self._X_fit.shape and np.allclose(X, self._X_fit):
            return self.embedding_
        raise NotImplementedError("sklearn TSNE does not support out-of-sample transform.")

    def _run_tsne(self, X: np.ndarray) -> np.ndarray:
        perplexity = min(self.perplexity, max(1.0, (X.shape[0] - 1) / 3))
        model = SklearnTSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            init=self.init,
            metric=self.metric,
            random_state=self.random_state,
        )
        return model.fit_transform(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            **super().get_params(),
            "perplexity": self.perplexity,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "init": self.init,
            "metric": self.metric,
        }
