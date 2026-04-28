from itertools import count

import numpy as np
from scipy.stats import norm
from .base import Classification

class ProbitRegression(Classification):
  """
  Probit regression (standard normal CDF link) fit with gradient descent.
  """

  def __init__(
    self,
    learning_rate: float = 1e-3,
    eps: float = 1e-6,
    max_iter: int | None = None,
    min_iter: int = 100,
    early_stopping_patience: int = 100,
    early_stopping_tol: float = 1e-12,
    class_weight: str | dict | None = None,
  ):
    super().__init__()
    self.class_weight = class_weight
    self.learning_rate = learning_rate
    self.eps = eps
    self.max_iter = max_iter
    self.min_iter = min_iter
    self.early_stopping_patience = early_stopping_patience
    self.early_stopping_tol = early_stopping_tol

  @staticmethod
  def _phi_cdf(z: np.ndarray) -> np.ndarray:
    """Cumulative distribution function of standard normal."""
    return norm.cdf(np.clip(z, -30, 30))

  @staticmethod
  def _phi_pdf(z: np.ndarray) -> np.ndarray:
    """Probability density function of standard normal."""
    return norm.pdf(np.clip(z, -30, 30))

  def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
  ) -> None:
    self.classes_ = np.unique(y)
    if len(self.classes_) != 2:
      raise ValueError(f"ProbitRegression is binary-only; got {len(self.classes_)} classes.")

    n_samples, n_features = X.shape
    y_binary = (y == self.classes_[1]).astype(float)

    if getattr(self, 'class_weight', None) is None:
      self.sample_weight_ = np.ones(n_samples)
    else:
      self.sample_weight_ = np.zeros(n_samples)
      for idx, cls in enumerate(self.classes_):
        mask = (y == cls)
        count = np.sum(mask)
        if isinstance(self.class_weight, dict):
          w = self.class_weight.get(cls, 1.0)
        elif self.class_weight == 'balanced':
          w = n_samples / (2.0 * count) if count > 0 else 0.0
        elif self.class_weight == 'proportional':
          w = count / n_samples
        else:
          raise ValueError(f"Unknown class_weight: {self.class_weight}")
        self.sample_weight_[mask] = w
      self.sample_weight_ *= n_samples / np.sum(self.sample_weight_)

    self.theta = np.zeros(n_features + 1)
    X_aug = self._augment(X)
    self.loss_history_ = []

    self._fit_gradient_descent(X_aug, y_binary)

  def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features = X.shape
    w = self.sample_weight_
    prev_loss = np.inf
    stagnation = 0
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for i in iterations:
      z = X @ self.theta
      cdf_z = self._phi_cdf(z)
      pdf_z = self._phi_pdf(z)

      denom = np.maximum(cdf_z * (1.0 - cdf_z), 1e-15)
      grad = (1.0 / n_samples) * (X.T @ (w * pdf_z * (cdf_z - y) / denom))
      
      update = self.learning_rate * grad
      self.theta -= update
      
      param_change = float(np.linalg.norm(update))
      grad_norm = float(np.linalg.norm(grad))
      
      loss = -np.mean(w * (y * np.log(cdf_z + 1e-15) + (1 - y) * np.log(1 - cdf_z + 1e-15)))
      self.loss_history_.append(loss)
      loss_change = abs(prev_loss - loss)
      if i + 1 >= self.min_iter and loss_change < self.early_stopping_tol:
        stagnation += 1
      else:
        stagnation = 0
      
      if i % 1000 == 0:
        print(f"Iteration {i}: Loss {loss:.4f}")
      if (
        i + 1 >= self.min_iter
        and param_change < self.eps
        and grad_norm < self.eps
        and loss_change < self.eps
      ):
        break
      if (
        self.early_stopping_patience > 0
        and stagnation >= self.early_stopping_patience
      ):
        break
      prev_loss = loss
    else:
      pass

  def predict(self, X: np.ndarray) -> np.ndarray:
    if self.theta is None or self.classes_ is None:
      raise RuntimeError("Call fit before predict.")
    probs = self.predict_proba(X)[:, 1]
    return self.classes_[(probs >= 0.5).astype(int)]

  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Return (n_samples, 2) with columns [P(y=class_0), P(y=class_1)]."""
    if self.theta is None:
      raise RuntimeError("Call fit before predict_proba.")
    X_aug = self._augment(X)
    p1 = self._phi_cdf(X_aug @ self.theta)
    return np.column_stack([1 - p1, p1])
