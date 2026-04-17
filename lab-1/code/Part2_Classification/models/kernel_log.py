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
    max_iter: int | None = 5000,
  ):
    super().__init__()
    self.kernel_type = kernel
    self.gamma = gamma
    self.lam = lam
    self.learning_rate = learning_rate
    self.eps = eps
    self.max_iter = max_iter
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
    
    # Compute Gram matrix
    K = self._get_kernel(X, X)
    
    # Initialize coefficients
    self.alpha = np.zeros(n_samples)
    self.b = 0.0
    
    i = 0
    while True:
      z = K @ self.alpha + self.b
      probs = self._sigmoid(z)
      
      # Compute gradients
      # dJ/d_alpha = K @ (1/n * (probs - y) + lam * alpha)
      # dJ/d_b = 1/n * sum(probs - y)
      err = probs - y_binary
      grad_alpha = K @ (err / n_samples + self.lam * self.alpha)
      grad_b = np.mean(err)
      
      # Updates
      update_alpha = self.learning_rate * grad_alpha
      update_b = self.learning_rate * grad_b
      
      self.alpha -= update_alpha
      self.b -= update_b
      
      update_norm = float(np.linalg.norm(update_alpha) + abs(update_b))
      
      if i % 500 == 0:
        # Cross-entropy loss + L2 regularization
        log_loss = -np.mean(y_binary * np.log(probs + 1e-15) + (1 - y_binary) * np.log(1 - probs + 1e-15))
        reg_loss = 0.5 * self.lam * (self.alpha @ K @ self.alpha)
        print(f"Iteration {i}: Loss {log_loss + reg_loss:.4f}")
        
      if update_norm < self.eps:
        print(f"Converged at iteration {i}: update norm {update_norm:.6e} < eps {self.eps:.6e}")
        break
      if self.max_iter is not None and i + 1 >= self.max_iter:
        print(f"Stopped at iteration {i + 1}: reached max_iter={self.max_iter} before convergence.")
        break
      i += 1

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
