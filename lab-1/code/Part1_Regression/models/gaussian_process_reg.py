import numpy as np

from .base import Regression


class GaussianProcessRegression(Regression):
    """Gaussian Process Regression with an RBF kernel (from scratch)."""

    def __init__(
      self,
      length_scale: float = 1.0,
      signal_variance: float = 1.0,
      noise_variance: float = 1e-2,
      jitter: float = 1e-10,
    ) -> None:
      self.length_scale = length_scale
      self.signal_variance = signal_variance
      self.noise_variance = noise_variance
      self.jitter = jitter

      self.X_train_: np.ndarray | None = None
      self.alpha_: np.ndarray | None = None
      self.L_: np.ndarray | None = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
      X1 = np.asarray(X1, dtype=float)
      X2 = np.asarray(X2, dtype=float)

      sq_dist = (
        np.sum(X1**2, axis=1, keepdims=True)
        + np.sum(X2**2, axis=1)
        - 2.0 * X1 @ X2.T
      )
      sq_dist = np.maximum(sq_dist, 0.0)
      return self.signal_variance * np.exp(
        -sq_dist / (2.0 * self.length_scale * self.length_scale)
      )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
      K = self._rbf_kernel(X, X)
      n = X.shape[0]
      K += (self.noise_variance + self.jitter) * np.eye(n)

      L = np.linalg.cholesky(K)
      alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

      self.X_train_ = X
      self.L_ = L
      self.alpha_ = alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
      if self.X_train_ is None or self.alpha_ is None:
        raise RuntimeError("Call fit before predict.")

      X = np.asarray(X, dtype=float)
      K_star = self._rbf_kernel(X, self.X_train_)
      return K_star @ self.alpha_

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      if self.X_train_ is None or self.alpha_ is None or self.L_ is None:
        raise RuntimeError("Call fit before predict_with_uncertainty.")

      X = np.asarray(X, dtype=float)
      K_star = self._rbf_kernel(X, self.X_train_)
      mean = K_star @ self.alpha_

      v = np.linalg.solve(self.L_, K_star.T)
      K_xx = self._rbf_kernel(X, X)
      var = np.diag(K_xx) - np.sum(v * v, axis=0)
      var = np.maximum(var, 0.0)
      std = np.sqrt(var)
      return mean, std
