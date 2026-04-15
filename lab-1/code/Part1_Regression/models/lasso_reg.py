import numpy as np

from .base import Regression


class LassoRegression(Regression):
    """Lasso Regression (L1) via Coordinate Descent with Soft-Thresholding.

    Minimises:
        (1/n) ||y - Xw - b||² + α ||w||₁

    The bias term ``b`` is updated as the mean residual and is never
    penalised.

    Parameters
    ----------
    alpha : float
        L1 regularization strength.
    max_iter : int
        Maximum coordinate-descent passes over all features.
    tol : float
        Stop early when the maximum absolute coefficient change in a pass
        is below this threshold.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    # Soft-thresholding operator  S(ρ, λ)
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_threshold(rho: float, lam: float) -> float:
        return float(np.sign(rho) * max(abs(rho) - lam, 0.0))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        n, d = X.shape
        self.coef_      = np.zeros(d)
        self.intercept_ = float(np.mean(y))

        # Pre-compute column squared norms — constant across iterations
        col_norms_sq = np.sum(X ** 2, axis=0) / n   # (d,)

        for _ in range(self.max_iter):
            max_delta = 0.0
            for j in range(d):
                # Partial residual: remove contribution of feature j
                r_j = y - self.intercept_ - X @ self.coef_ + X[:, j] * self.coef_[j]
                rho_j = float(X[:, j] @ r_j) / n

                old = self.coef_[j]
                # Normalised soft-threshold update
                z_j = col_norms_sq[j]
                if z_j < 1e-12:
                    self.coef_[j] = 0.0
                else:
                    self.coef_[j] = self._soft_threshold(rho_j, self.alpha / 2.0) / z_j

                max_delta = max(max_delta, abs(self.coef_[j] - old))

            # Update bias (never regularised)
            self.intercept_ = float(np.mean(y - X @ self.coef_))

            if max_delta < self.tol:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before predict.")
        return X @ self.coef_ + self.intercept_
