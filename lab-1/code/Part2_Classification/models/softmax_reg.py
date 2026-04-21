import numpy as np

from .base import Classification

class SoftmaxRegression(Classification):
  """
  Multinomial (softmax) regression fit with gradient descent until parameter
  updates are below ``eps`` (convex problem; unique optimum under full rank).
  """

  def __init__(
    self,
    learning_rate: float = 1e-4,
    eps: float = 1e-6,
    max_iter: int | None = None,
    prior_precision: float = 1.0,
    penalize_bias: bool = False,
    class_weight: str | dict | None = None,
  ):
    super().__init__()
    self.class_weight = class_weight
    self.learning_rate = learning_rate
    self.eps = eps
    self.max_iter = max_iter
    self.prior_precision = prior_precision
    self.penalize_bias = penalize_bias
    self.posterior_cov: np.ndarray | None = None

  @staticmethod
  def _softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

  @staticmethod
  def _one_hot(y_idx: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=float)[y_idx]

  def _prior_mask(self, n_features_aug: int, n_classes: int) -> np.ndarray:
    mask = np.ones((n_features_aug, n_classes), dtype=float)
    if not self.penalize_bias:
      mask[-1, :] = 0.0
    return mask

  def fit(self, X: np.ndarray, y: np.ndarray, solver: str = "gradient_descent") -> None:
    self.classes_ = np.unique(y)
    n_samples, n_features = X.shape
    n_classes = len(self.classes_)
    y_idx = np.searchsorted(self.classes_, y)

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
          w = n_samples / (n_classes * count) if count > 0 else 0.0
        elif self.class_weight == 'proportional':
          w = count / n_samples
        else:
          raise ValueError(f"Unknown class_weight: {self.class_weight}")
        self.sample_weight_[mask] = w
      self.sample_weight_ *= n_samples / np.sum(self.sample_weight_)

    self.theta = np.zeros((n_features + 1, n_classes))
    X_aug = self._augment(X)
    y_encoded = self._one_hot(y_idx, n_classes)
    self.posterior_cov = None
    self.loss_history_ = []

    if solver == "gradient_descent":
      self._fit_gradient_descent(X_aug, y_encoded)
    elif solver == "laplace":
      self._fit_laplace(X_aug, y_encoded)
    else:
      raise ValueError(
        "Invalid solver: "
        f"{solver}. Valid solvers are 'gradient_descent' and 'laplace'."
      )

  def _fit_gradient_descent(self, X: np.ndarray, y_encoded: np.ndarray) -> None:
    n_samples = X.shape[0]
    w = self.sample_weight_[:, np.newaxis]
    i = 0
    while True:
      scores = X @ self.theta
      probs = self._softmax(scores)

      d_theta = (1.0 / n_samples) * (X.T @ (w * (probs - y_encoded)))
      update = self.learning_rate * d_theta
      self.theta -= update

      update_norm = float(np.linalg.norm(update))

      loss = -np.mean(w.ravel() * np.sum(y_encoded * np.log(probs + 1e-15), axis=1))
      self.loss_history_.append(loss)

      if i % 100 == 0:
        print(f"Iteration {i}: Loss {loss:.4f}")

      if update_norm < self.eps:
        print(f"Converged at iteration {i}: update norm {update_norm:.6e} < eps {self.eps:.6e}")
        break
      if self.max_iter is not None and i + 1 >= self.max_iter:
        print(f"Stopped at iteration {i + 1}: reached max_iter={self.max_iter} before convergence.")
        break
      i += 1

  def _fit_laplace(self, X: np.ndarray, y_encoded: np.ndarray) -> None:
    n_samples, n_features_aug = X.shape
    n_classes = y_encoded.shape[1]
    prior_mask = self._prior_mask(n_features_aug, n_classes)
    prior_diag = (self.prior_precision / n_samples) * prior_mask.ravel()
    w_weight = self.sample_weight_

    hessian = None
    i = 0
    while True:
      scores = X @ self.theta
      probs = self._softmax(scores)

      grad = (X.T @ (w_weight[:, np.newaxis] * (probs - y_encoded))) / n_samples
      grad += (self.prior_precision / n_samples) * (prior_mask * self.theta)
      grad_vec = grad.ravel()

      dk = n_features_aug * n_classes
      hessian = np.zeros((dk, dk), dtype=float)
      for idx in range(n_samples):
        p = probs[idx]
        s = np.diag(p) - np.outer(p, p)
        xx = np.outer(X[idx], X[idx])
        hessian += w_weight[idx] * np.kron(xx, s) / n_samples

      hessian += np.diag(prior_diag)
      hessian += 1e-8 * np.eye(dk)

      step = np.linalg.solve(hessian, grad_vec).reshape(n_features_aug, n_classes)
      self.theta -= step

      step_norm = float(np.linalg.norm(step))
      
      map_loss = -np.mean(w_weight * np.sum(y_encoded * np.log(probs + 1e-15), axis=1))
      prior_term = 0.5 * (self.prior_precision / n_samples) * np.sum(
        prior_mask * (self.theta ** 2)
      )
      self.loss_history_.append(map_loss + prior_term)
      
      if i % 100 == 0:
        print(f"Iteration {i}: MAP objective {map_loss + prior_term:.4f}")
      if step_norm < self.eps:
        print(f"Converged at iteration {i}: step norm {step_norm:.6e} < eps {self.eps:.6e}")
        break
      i += 1

    if hessian is None:
      raise RuntimeError("Laplace fitting failed before computing Hessian.")
    self.posterior_cov = np.linalg.pinv(hessian)
    self.posterior_cov = 0.5 * (self.posterior_cov + self.posterior_cov.T)

  def predict(self, X: np.ndarray) -> np.ndarray:
    if self.theta is None or self.classes_ is None:
      raise RuntimeError("Call fit before predict.")
    probs = self.predict_proba(X, predictive="map")
    return self.classes_[np.argmax(probs, axis=1)]

  def predict_proba(
    self,
    X: np.ndarray,
    predictive: str = "map",
    n_samples: int = 200,
  ) -> np.ndarray:
    """Return (n_samples, n_classes) softmax probabilities."""
    if self.theta is None:
      raise RuntimeError("Call fit before predict_proba.")
    X_aug = self._augment(X)

    if predictive not in {"auto", "map", "laplace"}:
      raise ValueError("predictive must be one of {'auto', 'map', 'laplace'}.")

    use_laplace = predictive == "laplace"
    if not use_laplace or self.posterior_cov is None:
      scores = X_aug @ self.theta
      return self._softmax(scores)

    if n_samples < 1:
      raise ValueError("n_samples must be at least 1 for Laplace prediction.")

    theta_vec = self.theta.ravel()
    sampled_theta = np.random.multivariate_normal(theta_vec, self.posterior_cov, size=n_samples)
    sampled_theta = sampled_theta.reshape(n_samples, self.theta.shape[0], self.theta.shape[1])

    scores = np.einsum("nd,sdk->snk", X_aug, sampled_theta)
    scores = scores - np.max(scores, axis=2, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=2, keepdims=True)
    return np.mean(probs, axis=0)
