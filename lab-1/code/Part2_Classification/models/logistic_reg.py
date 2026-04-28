import numpy as np
from itertools import count

from .base import Classification


class LogisticRegression(Classification):
  """
  Binary logistic regression (sigmoid + binary cross-entropy) fit with
  gradient descent until parameter updates are below ``eps``.
  """

  def __init__(
    self,
    learning_rate: float = 1e-2,
    eps: float = 1e-4,
    max_iter: int | None = None,
    min_iter: int = 100,
    early_stopping_patience: int = 100,
    early_stopping_tol: float = 1e-12,
    prior_precision: float = 1.0,
    penalize_bias: bool = False,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
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
    self.prior_precision = prior_precision
    self.penalize_bias = penalize_bias
    self.l1_penalty = l1_penalty
    self.l2_penalty = l2_penalty
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
    self.posterior_cov = None
    self.loss_history_ = []

    if solver == 'gradient_descent':
      self._fit_gradient_descent(X_aug, y_binary)
    elif solver == 'newton_raphson' or solver == 'IRLS':
      self._fit_newton_raphson(X_aug, y_binary)
    elif solver == 'laplace_approximation' or solver == 'laplace':
      self._fit_laplace(X_aug, y_binary)
    elif solver == 'l1':
      self._fit_l1(X_aug, y_binary)
    elif solver == 'l2':
      self._fit_l2(X_aug, y_binary)
    elif solver == 'elastic_net':
      self._fit_elastic_net(X_aug, y_binary)
    else:
      raise ValueError(f"Invalid solver: {solver}. Valid solvers are 'gradient_descent', 'newton_raphson', 'IRLS', 'laplace', 'l1', 'l2', and 'elastic_net'.")

  def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features = X.shape
    self.theta = np.zeros(n_features)
    w = self.sample_weight_
    prev_loss = np.inf
    stagnation = 0
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for i in iterations:
      z = X @ self.theta
      probs = self._sigmoid(z)
      d_theta = (1.0 / n_samples) * (X.T @ (w * (probs - y)))
      update = self.learning_rate * d_theta
      self.theta -= update
      param_change = float(np.linalg.norm(update))
      grad_norm = float(np.linalg.norm(d_theta))
      
      loss = -np.mean(w * (y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15)))
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
      if self.early_stopping_patience > 0 and stagnation >= self.early_stopping_patience:
        break
      prev_loss = loss
    else:
      pass

  def _fit_l1(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features = X.shape
    self.theta = np.zeros(n_features)
    prior_mask = self._prior_mask(n_features)
    w = self.sample_weight_
    prev_loss = np.inf
    stagnation = 0
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for i in iterations:
      z = X @ self.theta
      probs = self._sigmoid(z)
      d_theta = (1.0 / n_samples) * (X.T @ (w * (probs - y))) + self.l1_penalty * (prior_mask * np.sign(self.theta))
      update = self.learning_rate * d_theta
      self.theta -= update
      param_change = float(np.linalg.norm(update))
      grad_norm = float(np.linalg.norm(d_theta))
      
      loss = -np.mean(w * (y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15)))
      loss += self.l1_penalty * np.sum(prior_mask * np.abs(self.theta))
      self.loss_history_.append(loss)
      loss_change = abs(prev_loss - loss)
      if i + 1 >= self.min_iter and loss_change < self.early_stopping_tol:
        stagnation += 1
      else:
        stagnation = 0
      
      if i % 1000 == 0:
        print(f"Iteration {i} (L1): Loss {loss:.4f}")
      if (
        i + 1 >= self.min_iter
        and param_change < self.eps
        and grad_norm < self.eps
        and loss_change < self.eps
      ):
        break
      if self.early_stopping_patience > 0 and stagnation >= self.early_stopping_patience:
        break
      prev_loss = loss
    else:
      pass

  def _fit_l2(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features = X.shape
    self.theta = np.zeros(n_features)
    prior_mask = self._prior_mask(n_features)
    w = self.sample_weight_
    prev_loss = np.inf
    stagnation = 0
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for i in iterations:
      z = X @ self.theta
      probs = self._sigmoid(z)
      d_theta = (1.0 / n_samples) * (X.T @ (w * (probs - y))) + self.l2_penalty * (prior_mask * self.theta)
      update = self.learning_rate * d_theta
      self.theta -= update
      param_change = float(np.linalg.norm(update))
      grad_norm = float(np.linalg.norm(d_theta))
      
      loss = -np.mean(w * (y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15)))
      loss += 0.5 * self.l2_penalty * np.sum(prior_mask * (self.theta ** 2))
      self.loss_history_.append(loss)
      loss_change = abs(prev_loss - loss)
      if i + 1 >= self.min_iter and loss_change < self.early_stopping_tol:
        stagnation += 1
      else:
        stagnation = 0
      
      if i % 1000 == 0:
        print(f"Iteration {i} (L2): Loss {loss:.4f}")
      if (
        i + 1 >= self.min_iter
        and param_change < self.eps
        and grad_norm < self.eps
        and loss_change < self.eps
      ):
        break
      if self.early_stopping_patience > 0 and stagnation >= self.early_stopping_patience:
        break
      prev_loss = loss
    else:
      pass

  def _fit_elastic_net(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features = X.shape
    self.theta = np.zeros(n_features)
    prior_mask = self._prior_mask(n_features)
    w = self.sample_weight_
    prev_loss = np.inf
    stagnation = 0
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for i in iterations:
      z = X @ self.theta
      probs = self._sigmoid(z)
      d_theta = (1.0 / n_samples) * (X.T @ (w * (probs - y)))
      d_theta += self.l2_penalty * (prior_mask * self.theta)
      d_theta += self.l1_penalty * (prior_mask * np.sign(self.theta))
      update = self.learning_rate * d_theta
      self.theta -= update
      param_change = float(np.linalg.norm(update))
      grad_norm = float(np.linalg.norm(d_theta))
      
      loss = -np.mean(w * (y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15)))
      loss += 0.5 * self.l2_penalty * np.sum(prior_mask * (self.theta ** 2))
      loss += self.l1_penalty * np.sum(prior_mask * np.abs(self.theta))
      self.loss_history_.append(loss)
      loss_change = abs(prev_loss - loss)
      if i + 1 >= self.min_iter and loss_change < self.early_stopping_tol:
        stagnation += 1
      else:
        stagnation = 0
      
      if i % 1000 == 0:
        print(f"Iteration {i} (Elastic Net): Loss {loss:.4f}")
      if (
        i + 1 >= self.min_iter
        and param_change < self.eps
        and grad_norm < self.eps
        and loss_change < self.eps
      ):
        break
      if self.early_stopping_patience > 0 and stagnation >= self.early_stopping_patience:
        break
      prev_loss = loss
    else:
      pass

  def _fit_newton_raphson(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features_aug = X.shape
    self.theta = np.zeros(n_features_aug)
    sw = self.sample_weight_
    
    prev_loss = np.inf
    stagnation = 0
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for i in iterations:
      logits = X @ self.theta
      probs = self._sigmoid(logits)
      w = sw * probs * (1.0 - probs)

      grad = (X.T @ (sw * (probs - y))) / n_samples
      hessian = (X.T * w) @ X / n_samples
      
      hessian += 1e-8 * np.eye(n_features_aug)
      
      try:
          step = np.linalg.solve(hessian, grad)
      except np.linalg.LinAlgError:
          step = np.linalg.pinv(hessian) @ grad
          
      self.theta -= step
      param_change = float(np.linalg.norm(step))
      grad_norm = float(np.linalg.norm(grad))
      
      loss = -np.mean(sw * (y * np.log(probs + 1e-15) + (1.0 - y) * np.log(1.0 - probs + 1e-15)))
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
      if self.early_stopping_patience > 0 and stagnation >= self.early_stopping_patience:
        break
      prev_loss = loss
    else:
      pass

  def _fit_laplace(self, X: np.ndarray, y: np.ndarray) -> None:
    n_samples, n_features_aug = X.shape
    self.theta = np.zeros(n_features_aug)
    prior_mask = self._prior_mask(n_features_aug)
    reg = self.prior_precision / n_samples
    hessian = None
    sw = self.sample_weight_

    prev_loss = np.inf
    stagnation = 0
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for i in iterations:
      logits = X @ self.theta
      probs = self._sigmoid(logits)
      w = sw * probs * (1.0 - probs)

      grad = (X.T @ (sw * (probs - y))) / n_samples + reg * (prior_mask * self.theta)
      hessian = (X.T * w) @ X / n_samples
      hessian += reg * np.diag(prior_mask)
      hessian += 1e-8 * np.eye(n_features_aug)

      step = np.linalg.solve(hessian, grad)
      self.theta -= step

      param_change = float(np.linalg.norm(step))
      grad_norm = float(np.linalg.norm(grad))
      
      map_loss = -np.mean(sw * (y * np.log(probs + 1e-15) + (1.0 - y) * np.log(1.0 - probs + 1e-15)))
      prior_term = 0.5 * reg * np.sum(prior_mask * (self.theta ** 2))
      objective = map_loss + prior_term
      self.loss_history_.append(objective)
      loss_change = abs(prev_loss - objective)
      if i + 1 >= self.min_iter and loss_change < self.early_stopping_tol:
        stagnation += 1
      else:
        stagnation = 0
      
      if i % 50 == 0:
        print(f"Iteration {i}: MAP objective {objective:.4f}")
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

    if hessian is None:
      raise RuntimeError("Laplace fitting failed before computing Hessian.")
    self.posterior_cov = np.linalg.pinv(hessian)
    self.posterior_cov = 0.5 * (self.posterior_cov + self.posterior_cov.T) # Ensure symmetry

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
      var_logits = np.einsum("ij,jk,ik->i", X_aug, self.posterior_cov, X_aug) # sigma^2 = x^T Sigma x
      var_logits = np.maximum(var_logits, 0.0)
      scaled_logits = mean_logits / np.sqrt(1.0 + (np.pi / 8.0) * var_logits)
      p1 = self._sigmoid(scaled_logits)
    else:
      p1 = self._sigmoid(X_aug @ self.theta)

    return np.column_stack([1 - p1, p1])
