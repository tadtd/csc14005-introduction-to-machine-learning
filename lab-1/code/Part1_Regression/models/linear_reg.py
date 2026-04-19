import numpy as np

from .base import Regression


class LinearRegression(Regression):
    """Linear Regression via Normal Equation, Batch GD, or Mini-batch GD.

    Parameters
    ----------
    solver : {'normal', 'gradient_descent', 'batch_gradient_descent', 'mini_batch_gradient_descent'}
        ``'normal'``  — closed-form solution via ``np.linalg.lstsq``.
        ``'gradient_descent'`` and ``'batch_gradient_descent'`` — iterative
        full-batch GD until convergence or ``max_iter`` is reached.
        ``'mini_batch_gradient_descent'`` — mini-batch GD over shuffled
        batches each epoch.
    learning_rate : float
        Step size for gradient descent (ignored when ``solver='normal'``).
    eps : float
        Convergence threshold: stop GD when ``||Δθ||₂ < eps``.
    max_iter : int or None
        Hard epoch cap for GD. ``None`` means no cap.
    batch_size : int
        Mini-batch size used when ``solver='mini_batch_gradient_descent'``.
    shuffle : bool
        Whether to shuffle sample order at each epoch for mini-batch GD.
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
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.solver = solver
        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.loss_history_: list[float] = []
        self.theta_: np.ndarray | None = None
        self.n_iter_: int = 0
        self.final_update_norm_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        if self.solver == "normal":
            self._fit_normal_equation(X, y)
        elif self.solver in {"gradient_descent", "batch_gradient_descent"}:
            self._fit_batch_gradient_descent(X, y)
        elif self.solver == "mini_batch_gradient_descent":
            self._fit_mini_batch_gradient_descent(X, y)
        else:
            raise ValueError(
                f"Unknown solver {self.solver!r}. "
                "Choose 'normal', 'gradient_descent', "
                "'batch_gradient_descent', or 'mini_batch_gradient_descent'."
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

    def _init_theta(self, d_aug: int) -> np.ndarray:
        """Initialise theta with optional reproducibility control."""
        if self.random_state is None:
            return np.random.randn(d_aug) * 0.01

        rng = np.random.default_rng(self.random_state)
        return rng.normal(loc=0.0, scale=0.01, size=d_aug)

    def _fit_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Full-batch GD with convergence check on update norm."""
        X_b = self._augment(X)
        n, d_aug = X_b.shape

        theta = self._init_theta(d_aug)
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
            self.final_update_norm_ = update_norm

            if update_norm < self.eps:
                break

        self.theta_ = np.asarray(theta, dtype=float).reshape(-1)

    def _fit_mini_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Mini-batch GD using optional per-epoch shuffling."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        X_b = self._augment(X)
        n, d_aug = X_b.shape
        theta = self._init_theta(d_aug)

        max_epochs = self.max_iter if self.max_iter is not None else 1_000_000
        self.loss_history_ = []
        self.final_update_norm_ = None
        self.n_iter_ = 0

        rng = np.random.default_rng(self.random_state) if self.random_state is not None else None

        for _ in range(max_epochs):
            if self.shuffle:
                if rng is None:
                    order = np.random.permutation(n)
                else:
                    order = rng.permutation(n)
            else:
                order = np.arange(n)

            max_update_norm = 0.0

            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                X_batch = X_b[idx]
                y_batch = y[idx]

                error_batch = X_batch @ theta - y_batch
                grad = (2.0 / len(idx)) * (X_batch.T @ error_batch)
                delta = self.learning_rate * grad
                theta -= delta

                update_norm = float(np.linalg.norm(delta))
                max_update_norm = max(max_update_norm, update_norm)
                self.n_iter_ += 1

            full_error = X_b @ theta - y
            self.loss_history_.append(float(np.mean(full_error ** 2)))
            self.final_update_norm_ = max_update_norm

            if max_update_norm < self.eps:
                break

        self.theta_ = np.asarray(theta, dtype=float).reshape(-1)
