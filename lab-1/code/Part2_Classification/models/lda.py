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
    self.components_: np.ndarray | None = None
    self.eigenvalues_: np.ndarray | None = None

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
    self._compute_projection_basis(X=X, y=y)

    self.theta = np.zeros((n_features + 1, n_classes))
    for idx in range(n_classes):
      mu = self.means_[idx]
      w = np.linalg.solve(pooled, mu)
      self.theta[:-1, idx] = w
      self.theta[-1, idx] = -0.5 * (mu @ w) + np.log(self.priors_[idx])

  def _compute_projection_basis(
    self,
    X: np.ndarray,
    y: np.ndarray,
  ) -> None:
    if self.classes_ is None:
      raise RuntimeError("Call fit before computing projection.")

    n_features = X.shape[1]
    overall_mean = X.mean(axis=0)

    sw = np.zeros((n_features, n_features), dtype=float)
    sb = np.zeros((n_features, n_features), dtype=float)

    for idx, cls in enumerate(self.classes_):
      X_cls = X[y == cls]
      centered = X_cls - self.means_[idx]
      sw += centered.T @ centered

      mean_diff = (self.means_[idx] - overall_mean).reshape(-1, 1)
      sb += X_cls.shape[0] * (mean_diff @ mean_diff.T)

    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(sw) @ sb)
    order = np.argsort(eigvals.real)[::-1]
    self.eigenvalues_ = eigvals.real[order]
    self.components_ = eigvecs[:, order].real

  def transform(
    self,
    X: np.ndarray,
    n_components: int | None = None,
  ) -> np.ndarray:
    if self.components_ is None:
      raise RuntimeError("Call fit before transform.")

    if n_components is None:
      n_components = self.components_.shape[1]

    if n_components <= 0:
      raise ValueError("n_components must be positive.")

    n_components = min(n_components, self.components_.shape[1])
    return X @ self.components_[:, :n_components]

  def fit_transform(
    self,
    X: np.ndarray,
    y: np.ndarray,
    n_components: int | None = None,
  ) -> np.ndarray:
    self.fit(X=X, y=y)
    return self.transform(X=X, n_components=n_components)

  def explained_discriminative_ratio(
    self,
    n_components: int | None = None,
  ) -> np.ndarray:
    if self.eigenvalues_ is None:
      raise RuntimeError("Call fit before explained_discriminative_ratio.")

    eigenvalues = np.clip(self.eigenvalues_, a_min=0.0, a_max=None)
    total = np.maximum(eigenvalues.sum(), 1e-12)

    if n_components is None:
      n_components = eigenvalues.shape[0]

    if n_components <= 0:
      raise ValueError("n_components must be positive.")

    n_components = min(n_components, eigenvalues.shape[0])
    return eigenvalues[:n_components] / total

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
