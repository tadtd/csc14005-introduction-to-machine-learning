import numpy as np

from .base import Regression


class KernelRidgeRegression(Regression):
    """Kernel Ridge Regression with RBF or polynomial kernels.

    Dual formulation:
        α = (K + λI)⁻¹ y
        ŷ(x) = k(x)ᵀ α

    where K depends on the selected kernel.

    Parameters
    ----------
    alpha : float
        Ridge regularization strength λ.
    gamma : float
        Kernel scale parameter γ.
        - RBF: K(x, z) = exp(-γ ||x - z||^2)
        - Polynomial: K(x, z) = (γ x^T z + coef0)^degree
    kernel : {'rbf', 'poly'}
        Kernel function to use.
    degree : int
        Degree of the polynomial kernel (used when kernel='poly').
    coef0 : float
        Independent term in polynomial kernel (used when kernel='poly').
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        coef0: float = 1.0,
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0.")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0.")
        if degree <= 0:
            raise ValueError("degree must be a positive integer.")

        self.alpha = alpha
        self.gamma = gamma
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.dual_coef_: np.ndarray | None = None  
        self.X_fit_: np.ndarray | None = None
    
    # Kernel
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

    def _poly_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the (n1 x n2) polynomial kernel matrix."""
        return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        kernel = self.kernel.lower()
        if kernel == "rbf":
            return self._rbf_kernel(X1, X2)
        if kernel in {"poly", "polynomial"}:
            return self._poly_kernel(X1, X2)
        raise ValueError(
            f"Unknown kernel {self.kernel!r}. Choose 'rbf' or 'poly'."
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        self.X_fit_ = X.copy()
        K = self._kernel_matrix(X, X)
        self.dual_coef_ = np.linalg.solve(
            K + self.alpha * np.eye(len(y)), y
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.dual_coef_ is None or self.X_fit_ is None:
            raise RuntimeError("Call fit before predict.")
        K = self._kernel_matrix(X, self.X_fit_)
        return K @ self.dual_coef_
