from itertools import count

import numpy as np
from .base import Classification

class KernelLogisticRegression(Classification):
  """
  Kernel Logistic Regression (Dual Formulation) fit with gradient descent.
  Supports RBF and Linear kernels.
  """

  def __init__(
    self,
    kernel: str = 'rbf',
    gamma: float = 0.5,
    lam: float = 1e-4,
    learning_rate: float = 1e-2,
    eps: float = 1e-6,
    max_iter: int | None = None,
    min_iter: int = 100,
    early_stopping_patience: int = 100,
    early_stopping_tol: float = 1e-12,
    class_weight: str | dict | None = None,
  ):
    super().__init__()
    self.class_weight = class_weight
    self.kernel_type = kernel
    self.gamma = gamma
    self.lam = lam
    self.learning_rate = learning_rate
    self.eps = eps
    self.max_iter = max_iter
    self.min_iter = min_iter
    self.early_stopping_patience = early_stopping_patience
    self.early_stopping_tol = early_stopping_tol
    self.alpha = None
    self.b = 0.0
    self.X_train = None

  def _get_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute the Gram matrix between X1 and X2."""
    if self.kernel_type == 'rbf':
      # Using the identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2x.y
      sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
      return np.exp(-self.gamma * sq_dist)
    elif self.kernel_type == 'linear':
      return np.dot(X1, X2.T)
    else:
      raise ValueError(f"Unsupported kernel: {self.kernel_type}")

  @staticmethod
  def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

  def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
  ) -> None:
    self.classes_ = np.unique(y)
    if len(self.classes_) != 2:
      raise ValueError("KernelLogisticRegression is binary-only.")

    n_samples = X.shape[0]
    self.X_train = X
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
    
    # Compute Gram matrix
    K = self._get_kernel(X, X)
    
    # Initialize coefficients
    self.alpha = np.zeros(n_samples)
    self.b = 0.0
    self.loss_history_ = []
    w = self.sample_weight_
    
    prev_loss = np.inf
    stagnation = 0
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for i in iterations:
      z = K @ self.alpha + self.b
      probs = self._sigmoid(z)
      
      err = w * (probs - y_binary)

      grad_alpha = (K @ err) / n_samples + self.lam * (K @ self.alpha)
      grad_b = np.mean(err)
      
      update_alpha = self.learning_rate * grad_alpha
      update_b = self.learning_rate * grad_b
      
      self.alpha -= update_alpha
      self.b -= update_b
      
      param_change = float(np.sqrt(np.linalg.norm(update_alpha) ** 2 + float(update_b) ** 2))
      grad_norm = float(np.linalg.norm(grad_alpha) + abs(grad_b))
      
      log_loss = -np.mean(w * (y_binary * np.log(probs + 1e-15) + (1 - y_binary) * np.log(1 - probs + 1e-15)))
      reg_loss = 0.5 * self.lam * (self.alpha @ K @ self.alpha)
      objective = log_loss + reg_loss
      self.loss_history_.append(objective)
      loss_change = abs(prev_loss - objective)
      if i + 1 >= self.min_iter and loss_change < self.early_stopping_tol:
        stagnation += 1
      else:
        stagnation = 0

      if i % 1000 == 0:
        print(f"Iteration {i}: Loss {objective:.4f}")
      if (
        i + 1 >= self.min_iter
        and param_change < self.eps
        and grad_norm < self.eps
        and loss_change < self.eps
      ):
        break
      if self.early_stopping_patience > 0 and stagnation >= self.early_stopping_patience:
        break
      prev_loss = objective
    else:
      pass

  def predict(self, X: np.ndarray) -> np.ndarray:
    if self.alpha is None or self.classes_ is None:
      raise RuntimeError("Call fit before predict.")
    probs = self.predict_proba(X)
    return self.classes_[(probs >= 0.5).astype(int)]

  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Return probability of the positive class."""
    if self.alpha is None or self.X_train is None:
      raise RuntimeError("Call fit before predict_proba.")
    K_test = self._get_kernel(X, self.X_train)
    z = K_test @ self.alpha + self.b
    return self._sigmoid(z)
