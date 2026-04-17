import numpy as np

from .base import Regression


class PerceptronRegression(Regression):
    """Gradient-descent regression with L2 (weight-decay) regularization.

    Minimises:
        (1/n) ||y - Xw - b||²  +  α ||w||²

    Tracks per-iteration training loss (MSE + L2 term) in ``loss_history_``
    for plotting learning / loss curves.

    Parameters
    ----------
    learning_rate : float
    max_iter : int
    alpha : float
        L2 regularization coefficient.
    random_state : int | None
        Optional seed for reproducible initialization. If None, uses the
        global NumPy RNG state.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 5_000,
        alpha: float = 0.01,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.random_state = random_state
        self.loss_history_: list[float] = []
        self.theta_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        X_b = self._augment(X)
        n, d_aug = X_b.shape
        if self.random_state is None:
            theta = np.random.randn(d_aug) * 0.01
        else:
            rng = np.random.default_rng(self.random_state)
            theta = rng.normal(loc=0.0, scale=0.01, size=d_aug)
        self.loss_history_ = []
        self.n_iter_ = 0

        reg_mask = np.ones(d_aug)
        reg_mask[-1] = 0.0  # never penalise bias

        for step in range(1, self.max_iter + 1):
            error = X_b @ theta - y

            grad = (2.0 / n) * (X_b.T @ error) + 2.0 * self.alpha * theta * reg_mask
            delta = self.learning_rate * grad
            theta -= delta

            # Track composite loss for convergence inspection
            train_loss = float(np.mean(error ** 2)) + self.alpha * float(np.sum((theta[:-1]) ** 2))
            self.loss_history_.append(train_loss)
            update_norm = float(np.linalg.norm(delta))
            self.n_iter_ = step

            if update_norm < self.tol:
                break

        self.theta_ = np.asarray(theta, dtype=float).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Call fit before predict.")
        return self._augment(X) @ self.theta_
