from __future__ import annotations

from typing import Any

import numpy as np

from .base import Regression


class FeatureMapRegressor(Regression):
    """Wrap a basis transformer and a base regression model.

    The transformer must expose ``fit``, ``transform`` and optionally
    ``get_feature_names_out``. The base model must follow the local
    ``Regression`` interface.
    """

    def __init__(self, transformer: Any, base_model: Regression) -> None:
        self.transformer = transformer
        self.base_model = base_model
        self.feature_names_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.transformer.fit(X, y)
        Z = self.transformer.transform(X)
        self.base_model.fit(Z, y, **kwargs)

        if hasattr(self.transformer, "get_feature_names_out"):
            try:
                names = self.transformer.get_feature_names_out()
            except TypeError:
                names = self.transformer.get_feature_names_out(None)
            self.feature_names_ = np.asarray(names, dtype=object)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.transformer.transform(np.asarray(X, dtype=float))
        return self.base_model.predict(Z)

    @property
    def coef_(self) -> np.ndarray:
        return self.base_model.coef_

    @property
    def intercept_(self) -> float:
        return self.base_model.intercept_
