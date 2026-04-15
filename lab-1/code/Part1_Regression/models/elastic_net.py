import numpy as np

from .base import Regression


class ElasticNetRegression(Regression):
    """Elastic Net Regression (L1 + L2) via Coordinate Descent.

    Minimises:
        (1/n) ||y - Xw - b||²  +  λ₁ ||w||₁  +  λ₂ ||w||²

    L2 term enters the coordinate update denominator (ridge-like shrinkage),
    while L1 applies soft-thresholding.  The bias ``b`` is not penalised.

    Parameters
    ----------
    lambda1 : float
        Weight for the L1 (Lasso) penalty.
    lambda2 : float
        Weight for the L2 (Ridge) penalty.
    max_iter : int
        Maximum coordinate-descent passes.
    tol : float
        Early-stopping threshold on max |Δcoef| per pass.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _soft_threshold(rho: float, lam: float) -> float:
        return float(np.sign(rho) * max(abs(rho) - lam, 0.0))

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        n, d = X.shape
        self.coef_      = np.zeros(d)
        self.intercept_ = float(np.mean(y))

        col_norms_sq = np.sum(X ** 2, axis=0) / n   # (d,)

        for _ in range(self.max_iter):
            max_delta = 0.0
            for j in range(d):
                r_j  = y - self.intercept_ - X @ self.coef_ + X[:, j] * self.coef_[j]
                rho_j = float(X[:, j] @ r_j) / n

                old = self.coef_[j]
                # Elastic-net update: soft-thresh numerator, ridge denominator
                z_j = col_norms_sq[j] + 2.0 * self.lambda2
                if z_j < 1e-12:
                    self.coef_[j] = 0.0
                else:
                    self.coef_[j] = self._soft_threshold(rho_j, self.lambda1 / 2.0) / z_j

                max_delta = max(max_delta, abs(self.coef_[j] - old))

            self.intercept_ = float(np.mean(y - X @ self.coef_))

            if max_delta < self.tol:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before predict.")
        return X @ self.coef_ + self.intercept_
