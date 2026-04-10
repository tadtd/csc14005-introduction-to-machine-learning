import numpy as np

from .base import Classification


class QDA(Classification):
  """
  Quadratic Discriminant Analysis: each class is Gaussian with its own covariance.
  Discriminant scores are quadratic in x; boundaries are quadric surfaces.
  """

  def __init__(self, reg: float = 1e-6):
    super().__init__()
    self.reg = reg
    self.priors_: np.ndarray | None = None
    self.means_: np.ndarray | None = None
    self.covariances_: list[np.ndarray] | None = None
    self.log_dets_: np.ndarray | None = None

  def fit(self, X: np.ndarray, y: np.ndarray) -> None:
    self.classes_ = np.unique(y)
    n_samples, n_features = X.shape
    n_classes = len(self.classes_)

    self.priors_ = np.zeros(n_classes)
    self.means_ = np.zeros((n_classes, n_features))
    self.covariances_ = []
    self.log_dets_ = np.zeros(n_classes)

    for idx, cls in enumerate(self.classes_):
      X_cls = X[y == cls]
      n_k = X_cls.shape[0]
      self.priors_[idx] = n_k / n_samples
      self.means_[idx] = X_cls.mean(axis=0)

      centered = X_cls - self.means_[idx]
      denom = max(n_k - 1, 1)
      cov = (centered.T @ centered) / denom
      cov += self.reg * np.eye(n_features)

      sign, log_det = np.linalg.slogdet(cov)
      if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")

      self.covariances_.append(cov)
      self.log_dets_[idx] = log_det

  def _decision_function(self, X: np.ndarray) -> np.ndarray:
    """
    Log posterior up to a class-independent constant in x:
    g_k(x) = -0.5 log|Sigma_k| - 0.5 (x-mu_k)^T Sigma_k^{-1} (x-mu_k) + log pi_k
    """
    if (
      self.classes_ is None
      or self.means_ is None
      or self.covariances_ is None
      or self.log_dets_ is None
      or self.priors_ is None
    ):
      raise RuntimeError("Call fit before predict.")

    n_samples = X.shape[0]
    n_classes = len(self.classes_)
    scores = np.zeros((n_samples, n_classes))

    for idx in range(n_classes):
      mu = self.means_[idx]
      cov = self.covariances_[idx]
      diff = X - mu
      # (x-mu)^T Sigma^{-1} (x-mu) = sum_j diff_ij * (solve(Sigma, diff.T))_{j,i}
      solved = np.linalg.solve(cov, diff.T)
      quad = np.sum(diff * solved.T, axis=1)
      scores[:, idx] = (
        -0.5 * self.log_dets_[idx] - 0.5 * quad + np.log(self.priors_[idx])
      )

    return scores

  def predict(self, X: np.ndarray) -> np.ndarray:
    scores = self._decision_function(X)
    return self.classes_[np.argmax(scores, axis=1)]

  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Posterior class probabilities from softmax over discriminant scores."""
    scores = self._decision_function(X)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

  def fisher_ratio(self) -> np.ndarray:
    """
    Per-feature Fisher ratio J_j = S_B(j) / S_W(j) from fitted parameters.
    S_B uses class means and priors; S_W uses the weighted average of
    per-class covariance diagonals.
    """
    if (
      self.means_ is None
      or self.priors_ is None
      or self.covariances_ is None
      or self.classes_ is None
    ):
      raise RuntimeError("Call fit before fisher_ratio.")
    mu_overall = self.priors_ @ self.means_
    sb = (self.priors_[:, None] * (self.means_ - mu_overall) ** 2).sum(axis=0)
    n_features = self.means_.shape[1]
    sw = np.zeros(n_features)
    for idx in range(len(self.classes_)):
      sw += self.priors_[idx] * np.diag(self.covariances_[idx])
    return sb / (sw + 1e-12)
