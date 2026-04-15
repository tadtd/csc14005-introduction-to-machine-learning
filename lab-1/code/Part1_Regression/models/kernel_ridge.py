import numpy as np

from .base import Regression


class KernelRidgeRegression(Regression):
    """Kernel Ridge Regression with an RBF (Gaussian) kernel.

    Dual formulation:
        α = (K + λI)⁻¹ y
        ŷ(x) = k(x)ᵀ α

    where  K_{ij} = exp(−γ ||xᵢ − xⱼ||²).

    Parameters
    ----------
    alpha : float
        Ridge regularization strength λ.
    gamma : float
        RBF bandwidth parameter γ.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dual_coef_: np.ndarray | None = None
        self.X_fit_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the (n1 × n2) RBF kernel matrix."""
        # ||x1 - x2||² = ||x1||² + ||x2||² - 2 x1·x2
        sq_dist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1)
            - 2.0 * X1 @ X2.T
        )
        # Clamp numerical negatives from floating-point arithmetic
        return np.exp(-self.gamma * np.maximum(sq_dist, 0.0))

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        self.X_fit_ = X.copy()
        K = self._rbf_kernel(X, X)
        self.dual_coef_ = np.linalg.solve(
            K + self.alpha * np.eye(len(y)), y
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.dual_coef_ is None or self.X_fit_ is None:
            raise RuntimeError("Call fit before predict.")
        K = self._rbf_kernel(X, self.X_fit_)
        return K @ self.dual_coef_
