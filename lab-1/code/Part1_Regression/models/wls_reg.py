import numpy as np

from .base import Regression
from .linear_reg import LinearRegression
from diagnostics import breusch_pagan_test, estimate_wls_weights


class WeightedLeastSquaresRegression(Regression):
    """Weighted Least Squares regression.

    Parameters
    ----------
    ridge
        Optional L2 stabilization on coefficients. The bias term is not
        penalized.
    """

    def __init__(self, ridge: float = 0.0) -> None:
        self.ridge = float(ridge)
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative.")
        self.theta_: np.ndarray | None = None
        self.sample_weight_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> None:  # type: ignore[override]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if sample_weight.shape[0] != X.shape[0]:
            raise ValueError("sample_weight must have one entry per sample.")
        if np.any(sample_weight <= 0):
            raise ValueError("sample_weight must be strictly positive.")

        X_b = self._augment(X)
        X_w = X_b * sample_weight[:, None]
        reg = self.ridge * np.eye(X_b.shape[1], dtype=float)
        reg[-1, -1] = 0.0

        A = X_b.T @ X_w + reg
        b = X_b.T @ (sample_weight * y)
        self.theta_ = np.linalg.solve(A, b)
        self.sample_weight_ = sample_weight

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        return self._augment(X) @ self.theta_


class FeasibleWeightedLeastSquaresRegression(Regression):
    """Feasible WLS with weights estimated from an initial OLS fit.

    The model first fits OLS on the training fold, estimates inverse-variance
    sample weights from residual dispersion, then refits a weighted least
    squares solution. This makes WLS comparable inside cross-validation where
    fold-specific weights must be re-estimated on each training split.
    """

    def __init__(
        self,
        ridge: float = 1e-8,
        *,
        n_bins: int = 10,
    ) -> None:
        self.ridge = float(ridge)
        self.n_bins = int(n_bins)
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive.")

        self.theta_: np.ndarray | None = None
        self.sample_weight_: np.ndarray | None = None
        self.ols_model_: LinearRegression | None = None
        self.bp_result_: dict[str, float] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        self.ols_model_ = LinearRegression(solver="normal")
        self.ols_model_.fit(X, y)
        fitted = self.ols_model_.predict(X)
        residuals = y - fitted

        self.bp_result_ = breusch_pagan_test(X, residuals)
        self.sample_weight_ = estimate_wls_weights(
            fitted,
            residuals,
            n_bins=self.n_bins,
        )

        inner = WeightedLeastSquaresRegression(ridge=self.ridge)
        inner.fit(X, y, sample_weight=self.sample_weight_)
        self.theta_ = inner.theta_.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        return self._augment(X) @ self.theta_
