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
    random_state : int
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 5_000,
        alpha: float = 0.01,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        np.random.seed(self.random_state)
        n, d = X.shape
        self.coef_      = np.random.randn(d) * 0.01
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.max_iter):
            error = X @ self.coef_ + self.intercept_ - y

            # Gradient of MSE + L2 penalty wrt w (bias not regularised)
            grad_coef = (2.0 / n) * (X.T @ error) + 2.0 * self.alpha * self.coef_
            grad_bias  = (2.0 / n) * float(np.sum(error))

            self.coef_      -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_bias

            # Track composite loss for convergence inspection
            train_loss = float(np.mean(error ** 2)) + self.alpha * float(np.sum(self.coef_ ** 2))
            self.loss_history_.append(train_loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before predict.")
        return X @ self.coef_ + self.intercept_
