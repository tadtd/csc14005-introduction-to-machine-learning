import numpy as np

from .base import Classification


class LogisticRegression(Classification):
  """
  Binary logistic regression (sigmoid + binary cross-entropy) fit with
  gradient descent until parameter updates are below ``eps``.
  """

  def __init__(
    self,
    learning_rate: float = 1e-4,
    eps: float = 1e-6,
    max_iter: int = 10_000,
    prior_precision: float = 1.0,
    penalize_bias: bool = False,
  ):
    super().__init__()
    self.learning_rate = learning_rate
    self.eps = eps
    self.max_iter = max_iter
    self.prior_precision = prior_precision
    self.penalize_bias = penalize_bias
    self.posterior_cov: np.ndarray | None = None

  @staticmethod
  def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

  def _prior_mask(self, n_features_aug: int) -> np.ndarray:
    mask = np.ones(n_features_aug, dtype=float)
    if not self.penalize_bias:
      mask[-1] = 0.0
    return mask

  def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
    solver: str = 'gradient_descent',
  ) -> None:
    self.classes_ = np.unique(y)
    if len(self.classes_) != 2:
      raise ValueError(f"LogisticRegression is binary-only; got {len(self.classes_)} classes. Use SoftmaxRegression for multiclass problems.")

    n_samples, n_features = X.shape
    y_binary = (y == self.classes_[1]).astype(float)

    self.theta = np.zeros(n_features + 1)
    X_aug = self._augment(X)
    self.posterior_cov = None

    if solver == 'gradient_descent':
      self._fit_gradient_descent(X_aug, y_binary)
    elif solver == 'newton_raphson' or solver == 'IRLS':
      self._fit_newton_raphson(X_aug, y_binary)
    elif solver == 'laplace':
      self._fit_laplace(X_aug, y_binary)
    else:
      raise ValueError(f"Invalid solver: {solver}. Valid solvers are 'gradient_descent', 'newton_raphson', 'IRLS', and 'laplace'.")

  def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features = X.shape
    self.theta = np.zeros(n_features)
    i = 0
    while i < self.max_iter:
      z = X @ self.theta
      probs = self._sigmoid(z)
      d_theta = (1.0 / n_samples) * (X.T @ (probs - y))
      update = self.learning_rate * d_theta
      self.theta -= update
      update_norm = float(np.linalg.norm(update))
      if i % 500 == 0:
        loss = -np.mean(y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15))
        print(f"Iteration {i}: Loss {loss:.4f}")
      if update_norm < self.eps:
        print(f"Converged at iteration {i}: update norm {update_norm:.6e} < eps {self.eps:.6e}")
        break
      i += 1

  def _fit_newton_raphson(self, X: np.ndarray, y: np.ndarray) -> None:
    raise NotImplementedError("Newton-Raphson method is not implemented yet.")

  def _fit_laplace(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features_aug = X.shape
    self.theta = np.zeros(n_features_aug)
    prior_mask = self._prior_mask(n_features_aug)
    reg = self.prior_precision / n_samples
    hessian = None

    for i in range(self.max_iter):
      logits = X @ self.theta
      probs = self._sigmoid(logits)
      w = probs * (1.0 - probs)

      grad = (X.T @ (probs - y)) / n_samples + reg * (prior_mask * self.theta)
      hessian = (X.T * w) @ X / n_samples
      hessian += reg * np.diag(prior_mask)
      hessian += 1e-8 * np.eye(n_features_aug)

      step = np.linalg.solve(hessian, grad)
      self.theta -= step

      step_norm = float(np.linalg.norm(step))
      if i % 50 == 0:
        map_loss = -np.mean(y * np.log(probs + 1e-15) + (1.0 - y) * np.log(1.0 - probs + 1e-15))
        prior_term = 0.5 * reg * np.sum(prior_mask * (self.theta ** 2))
        print(f"Iteration {i}: MAP objective {map_loss + prior_term:.4f}")
      if step_norm < self.eps:
        print(f"Converged at iteration {i}: step norm {step_norm:.6e} < eps {self.eps:.6e}")
        break

    if hessian is None:
      raise RuntimeError("Laplace fitting failed before computing Hessian.")
    self.posterior_cov = np.linalg.pinv(hessian)
    self.posterior_cov = 0.5 * (self.posterior_cov + self.posterior_cov.T)

  def predict(self, X: np.ndarray) -> np.ndarray:
    if self.theta is None or self.classes_ is None:
      raise RuntimeError("Call fit before predict.")
    probs = self.predict_proba(X)[:, 1]
    return self.classes_[(probs >= 0.5).astype(int)]

  def predict_proba(
    self,
    X: np.ndarray,
    predictive: str = "auto",
  ) -> np.ndarray:
    """Return (n_samples, 2) with columns [P(y=class_0), P(y=class_1)]."""
    if self.theta is None:
      raise RuntimeError("Call fit before predict_proba.")
    X_aug = self._augment(X)

    if predictive not in {"auto", "map", "laplace"}:
      raise ValueError("predictive must be one of {'auto', 'map', 'laplace'}.")

    use_laplace = predictive == "laplace" or (predictive == "auto" and self.posterior_cov is not None)

    if use_laplace and self.posterior_cov is not None:
      mean_logits = X_aug @ self.theta
      var_logits = np.einsum("ij,jk,ik->i", X_aug, self.posterior_cov, X_aug)
      var_logits = np.maximum(var_logits, 0.0)
      scaled_logits = mean_logits / np.sqrt(1.0 + (np.pi / 8.0) * var_logits)
      p1 = self._sigmoid(scaled_logits)
    else:
      p1 = self._sigmoid(X_aug @ self.theta)

    return np.column_stack([1 - p1, p1])
