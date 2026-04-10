import numpy as np
from .base import Classification


class LDA(Classification):
  """
  Linear Discriminant Analysis: Gaussian class-conditional densities with a
  shared (pooled) covariance matrix. Decision boundaries are linear in x.
  """

  def __init__(
    self,
    reg: float = 1e-6,
  ):
    super().__init__()
    self.reg = reg
    self.priors_: np.ndarray | None = None
    self.means_: np.ndarray | None = None
    self.covariance_: np.ndarray | None = None

  def fit(
    self, 
    X: np.ndarray, 
    y: np.ndarray,
  ) -> None:
    self.classes_ = np.unique(y)
    n_samples, n_features = X.shape
    n_classes = len(self.classes_)

    self.priors_ = np.zeros(n_classes)
    self.means_ = np.zeros((n_classes, n_features))

    for idx, cls in enumerate(self.classes_):
      mask = y == cls
      X_cls = X[mask]
      self.priors_[idx] = X_cls.shape[0] / n_samples
      self.means_[idx] = X_cls.mean(axis=0)

    pooled = np.zeros((n_features, n_features), dtype=float)
    for idx, cls in enumerate(self.classes_):
      X_cls = X[y == cls]
      centered = X_cls - self.means_[idx]
      pooled += centered.T @ centered

    pooled /= max(n_samples - n_classes, 1)
    pooled += self.reg * np.eye(n_features)

    self.covariance_ = pooled

    self.theta = np.zeros((n_features + 1, n_classes))
    for idx in range(n_classes):
      mu = self.means_[idx]
      w = np.linalg.solve(pooled, mu)
      self.theta[:-1, idx] = w
      self.theta[-1, idx] = -0.5 * (mu @ w) + np.log(self.priors_[idx])

  def _decision_function(
    self, 
    X: np.ndarray
  ) -> np.ndarray:
    """Discriminant scores delta_k(x) (equal to log posterior up to a constant in x)."""
    if self.theta is None:
      raise RuntimeError("Call fit before predict.")
    return self._augment(X) @ self.theta

  def predict(self, X: np.ndarray) -> np.ndarray:
    scores = self._decision_function(X)
    return self.classes_[np.argmax(scores, axis=1)]

  def predict_proba(
    self, 
    X: np.ndarray,
  ) -> np.ndarray:
    """Class probabilities from softmax over discriminant scores (Gaussian LDA posteriors)."""
    scores = self._decision_function(X)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

  def fisher_ratio(self) -> np.ndarray:
    """
    Per-feature Fisher ratio J_j = S_B(j) / S_W(j) from fitted parameters.
    S_B uses class means and priors; S_W uses the pooled covariance diagonal.
    """
    if (
      self.means_ is None
      or self.priors_ is None
      or self.covariance_ is None
    ):
      raise RuntimeError("Call fit before fisher_ratio.")
    mu_overall = self.priors_ @ self.means_
    sb = (self.priors_[:, None] * (self.means_ - mu_overall) ** 2).sum(axis=0)
    sw = np.diag(self.covariance_) + 1e-12
    return sb / sw
