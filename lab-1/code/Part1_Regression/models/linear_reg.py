import numpy as np

from .base import Regression


class LinearRegression(Regression):
    """Linear Regression via Normal Equation or Batch Gradient Descent.

    Parameters
    ----------
    solver : {'normal', 'gradient_descent'}
        ``'normal'``  — closed-form solution via ``np.linalg.lstsq``.
        ``'gradient_descent'`` — iterative batch GD until convergence or
        ``max_iter`` is reached.
    learning_rate : float
        Step size for gradient descent (ignored when ``solver='normal'``).
    eps : float
        Convergence threshold: stop GD when ``||Δθ||₂ < eps``.
    max_iter : int or None
        Hard iteration cap for GD.  ``None`` means no cap.
    random_state : int | None
        Optional seed for reproducible GD initialization. If None, uses the
        global NumPy RNG state (so notebook-level seeding is respected).
    """

    def __init__(
        self,
        solver: str = "normal",
        learning_rate: float = 0.01,
        eps: float = 1e-6,
        max_iter: int | None = 50_000,
        random_state: int | None = None,
    ) -> None:
        self.solver = solver
        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state
        self.loss_history_: list[float] = []
        self.theta_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        if self.solver == "normal":
            self._fit_normal_equation(X, y)
        elif self.solver == "gradient_descent":
            self._fit_gradient_descent(X, y)
        else:
            raise ValueError(
                f"Unknown solver {self.solver!r}. "
                "Choose 'normal' or 'gradient_descent'."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Call fit before predict.")
        return self._augment(X) @ self.theta_

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Closed-form: θ = (X̃ᵀX̃)⁻¹ X̃ᵀy  via lstsq for numerical safety."""
        X_b = self._augment(X)           # (n, d+1); bias appended last
        theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.theta_ = np.asarray(theta, dtype=float).reshape(-1)
        self.n_iter_ = 1

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Batch GD with convergence check on the parameter update norm."""
        X_b = self._augment(X)
        n, d_aug = X_b.shape
        
        # Standardized init: respect global seed unless random_state is explicit.
        if self.random_state is None:
            theta = np.random.randn(d_aug) * 0.01
        else:
            rng = np.random.default_rng(self.random_state)
            theta = rng.normal(loc=0.0, scale=0.01, size=d_aug)
        self.loss_history_ = []
        max_steps = self.max_iter if self.max_iter is not None else 1_000_000

        self.final_update_norm_ = None
        self.n_iter_ = 0

        for step in range(1, max_steps + 1):
            error = X_b @ theta - y
            grad = (2.0 / n) * (X_b.T @ error)
            delta = self.learning_rate * grad
            theta -= delta

            self.loss_history_.append(float(np.mean(error ** 2)))
            update_norm = float(np.linalg.norm(delta))
            self.n_iter_ = step

            if update_norm < self.eps:
                break

        self.theta_ = np.asarray(theta, dtype=float).reshape(-1)
