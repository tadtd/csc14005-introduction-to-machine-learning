import numpy as np

from .base import Regression


class RidgeRegression(Regression):
    """Ridge Regression (L2 regularization) — closed-form solution.

    Solves: θ = (X̃ᵀX̃ + α·Ĩ)⁻¹ X̃ᵀy
    where Ĩ is the identity with the bias entry zeroed out so the bias
    term is never penalised.

    Parameters
    ----------
    alpha : float
        L2 regularization strength (λ).  Larger values shrink
        coefficients more aggressively.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.theta_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        X_b = self._augment(X)          # (n, d+1); bias appended last
        d = X_b.shape[1]

        # Regularisation matrix: do not penalise the bias term (last column)
        reg = self.alpha * np.eye(d)
        reg[-1, -1] = 0.0

        # Use solve instead of pinv for better numerical stability
        A = X_b.T @ X_b + reg
        b = X_b.T @ y
        theta = np.linalg.solve(A, b)
        self.theta_ = np.asarray(theta, dtype=float).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Call fit before predict.")
        return self._augment(X) @ self.theta_
