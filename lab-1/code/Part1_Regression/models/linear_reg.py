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
    random_state : int
        Seed used to initialise GD weights.
    """

    def __init__(
        self,
        solver: str = "normal",
        learning_rate: float = 0.01,
        eps: float = 1e-6,
        max_iter: int | None = 10_000,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.solver = solver
        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state
        self.loss_history_: list[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before predict.")
        return X @ self.coef_ + self.intercept_

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Closed-form: θ = (X̃ᵀX̃)⁻¹ X̃ᵀy  via lstsq for numerical safety."""
        X_b = self._augment(X)           # (n, d+1); bias appended last
        theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.coef_ = theta[:-1]
        self.intercept_ = float(theta[-1])

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Batch GD with convergence check on the parameter update norm."""
        np.random.seed(self.random_state)
        n, d = X.shape
        self.coef_ = np.random.randn(d) * 0.01
        self.intercept_ = 0.0
        self.loss_history_ = []

        for i in range(self.max_iter if self.max_iter is not None else 10 ** 9):
            error = X @ self.coef_ + self.intercept_ - y          # (n,)
            grad_coef = (2.0 / n) * (X.T @ error)
            grad_bias = (2.0 / n) * float(np.sum(error))

            delta_coef = self.learning_rate * grad_coef
            delta_bias = self.learning_rate * grad_bias

            self.coef_      -= delta_coef
            self.intercept_ -= delta_bias

            self.loss_history_.append(float(np.mean(error ** 2)))

            update_norm = float(
                np.sqrt(np.sum(delta_coef ** 2) + delta_bias ** 2)
            )
            if update_norm < self.eps:
                break
