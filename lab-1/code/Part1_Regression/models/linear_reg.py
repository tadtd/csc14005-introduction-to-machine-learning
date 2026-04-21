import numpy as np
import time

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
        learning_rate_schedule: str = "constant",
        step_decay_rate: float = 0.5,
        step_decay_epochs: int = 100,
        min_learning_rate: float = 1e-6,
    ) -> None:
        self.solver = solver
        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.learning_rate_schedule = learning_rate_schedule
        self.step_decay_rate = step_decay_rate
        self.step_decay_epochs = step_decay_epochs
        self.min_learning_rate = min_learning_rate
        self.loss_history_: list[float] = []
        self.learning_rate_history_: list[float] = []
        self.theta_: np.ndarray | None = None
        self.n_iter_: int = 0
        self.final_update_norm_: float | None = None
        self.fit_time_seconds_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        start = time.perf_counter()
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
        self.fit_time_seconds_ = time.perf_counter() - start

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

    def _current_learning_rate(self, epoch: int, total_epochs: int) -> float:
        schedule = self.learning_rate_schedule.lower()
        base_lr = float(self.learning_rate)

        if schedule == "constant":
            return base_lr

        if schedule == "step_decay":
            if self.step_decay_epochs <= 0:
                raise ValueError("step_decay_epochs must be positive.")
            factor = self.step_decay_rate ** (epoch // self.step_decay_epochs)
            return max(base_lr * factor, self.min_learning_rate)

        if schedule == "cosine_annealing":
            total = max(total_epochs, 1)
            cosine = 0.5 * (1.0 + np.cos(np.pi * epoch / total))
            return max(self.min_learning_rate, self.min_learning_rate + (base_lr - self.min_learning_rate) * cosine)

        raise ValueError(
            "learning_rate_schedule must be one of "
            "{'constant', 'step_decay', 'cosine_annealing'}."
        )

    def _fit_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Full-batch GD with convergence check on update norm."""
        X_b = self._augment(X)
        n, d_aug = X_b.shape

        theta = self._init_theta(d_aug)
        self.loss_history_ = []
        self.learning_rate_history_ = []
        max_steps = self.max_iter if self.max_iter is not None else 1_000_000

        self.final_update_norm_ = None
        self.n_iter_ = 0

        for step in range(1, max_steps + 1):
            lr = self._current_learning_rate(step - 1, max_steps)
            error = X_b @ theta - y
            grad = (2.0 / n) * (X_b.T @ error)
            delta = lr * grad
            theta -= delta

            self.loss_history_.append(float(np.mean(error ** 2)))
            self.learning_rate_history_.append(lr)
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
        self.learning_rate_history_ = []
        self.final_update_norm_ = None
        self.n_iter_ = 0

        rng = np.random.default_rng(self.random_state) if self.random_state is not None else None

        for epoch in range(max_epochs):
            if self.shuffle:
                if rng is None:
                    order = np.random.permutation(n)
                else:
                    order = rng.permutation(n)
            else:
                order = np.arange(n)

            max_update_norm = 0.0
            lr = self._current_learning_rate(epoch, max_epochs)

            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                X_batch = X_b[idx]
                y_batch = y[idx]

                error_batch = X_batch @ theta - y_batch
                grad = (2.0 / len(idx)) * (X_batch.T @ error_batch)
                delta = lr * grad
                theta -= delta

                update_norm = float(np.linalg.norm(delta))
                max_update_norm = max(max_update_norm, update_norm)
                self.n_iter_ += 1

            full_error = X_b @ theta - y
            self.loss_history_.append(float(np.mean(full_error ** 2)))
            self.learning_rate_history_.append(lr)
            self.final_update_norm_ = max_update_norm

            if max_update_norm < self.eps:
                break

        self.theta_ = np.asarray(theta, dtype=float).reshape(-1)
