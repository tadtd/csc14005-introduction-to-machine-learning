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
        max_iter: int = 5_000,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.theta_: np.ndarray | None = None
        self.n_iter_: int = 0
        self.max_delta_: float | None = None

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
        initial_theta = kwargs.get("initial_theta")
        if initial_theta is None:
            theta = np.zeros(d + 1)
            theta[-1] = float(np.mean(y))
        else:
            theta = np.asarray(initial_theta, dtype=float).reshape(-1).copy()
            if theta.shape[0] != d + 1:
                raise ValueError("initial_theta must have shape (n_features + 1,).")
        coef = theta[:-1]

        # Pre-compute column squared norms — constant across iterations
        col_norms_sq = np.sum(X ** 2, axis=0) / n   # (d,)
        self.n_iter_ = 0

        for step in range(1, self.max_iter + 1):
            max_delta = 0.0
            for j in range(d):
                # Partial residual: remove contribution of feature j
                r_j = y - theta[-1] - X @ coef + X[:, j] * coef[j]
                rho_j = float(X[:, j] @ r_j) / n

                old = coef[j]
                # Normalised soft-threshold update
                z_j = col_norms_sq[j]
                if z_j < 1e-12:
                    coef[j] = 0.0
                else:
                    coef[j] = self._soft_threshold(rho_j, self.alpha / 2.0) / z_j

                max_delta = max(max_delta, abs(coef[j] - old))

            # Update bias (never regularised)
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
