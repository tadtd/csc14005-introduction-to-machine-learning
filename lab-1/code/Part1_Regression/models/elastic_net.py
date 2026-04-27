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
        max_iter: int = 1_000,
        tol: float = 1e-4,
    ) -> None:
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.tol = tol
        self.theta_: np.ndarray | None = None
        self.n_iter_: int = 0
        self.max_delta_: float | None = None

    @staticmethod
    def _soft_threshold(rho: float, lam: float) -> float:
        return float(np.sign(rho) * max(abs(rho) - lam, 0.0))

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        n, d = X.shape
        initial_theta = kwargs.get("initial_theta")
        if initial_theta is None:
            theta = np.zeros(d + 1)
            theta[-1] = float(np.mean(y))
        else:
            theta = np.asarray(initial_theta, dtype=float).reshape(-1).copy()
            if theta.shape[0] != d + 1:
                raise ValueError("initial_theta must have shape (n_features + 1,).")
        coef = theta[:-1]

        col_norms_sq = np.sum(X ** 2, axis=0) / n   # (d,)
        self.n_iter_ = 0

        for step in range(1, self.max_iter + 1):
            max_delta = 0.0
            for j in range(d):
                r_j = y - theta[-1] - X @ coef + X[:, j] * coef[j]
                rho_j = float(X[:, j] @ r_j) / n

                old = coef[j]
                # Elastic-net update: soft-thresh numerator, ridge denominator
                z_j = col_norms_sq[j] + self.lambda2
                if z_j < 1e-12:
                    coef[j] = 0.0
                else:
                    coef[j] = self._soft_threshold(rho_j, self.lambda1 / 2.0) / z_j

                max_delta = max(max_delta, abs(coef[j] - old))

            theta[-1] = float(np.mean(y - X @ coef))
            self.n_iter_ = step
            self.max_delta_ = max_delta

            if max_delta < self.tol:
                break

        self.theta_ = np.asarray(theta, dtype=float).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Call fit before predict.")
        return self._augment(X) @ self.theta_
