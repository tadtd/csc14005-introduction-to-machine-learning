import numpy as np

from .base import Classification


class GaussianNaiveBayes(Classification):
  """
  Gaussian Naive Bayes with diagonal class-conditional covariance.
  """

  def __init__(self, reg: float = 1e-6) -> None:
    super().__init__()
    self.reg = reg
    self.priors_: np.ndarray | None = None
    self.means_: np.ndarray | None = None
    self.vars_: np.ndarray | None = None

  def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
  ) -> None:
    self.classes_ = np.unique(y)
    n_samples, n_features = X.shape
    n_classes = len(self.classes_)

    self.priors_ = np.zeros(n_classes, dtype=float)
    self.means_ = np.zeros((n_classes, n_features), dtype=float)
    self.vars_ = np.zeros((n_classes, n_features), dtype=float)

    for class_idx, class_label in enumerate(self.classes_):
      X_class = X[y == class_label]
      self.priors_[class_idx] = X_class.shape[0] / n_samples
      self.means_[class_idx] = X_class.mean(axis=0)
      # Use population variance to avoid ddof=1 NaNs for tiny classes.
      self.vars_[class_idx] = X_class.var(axis=0, ddof=0)

    self.vars_ = np.maximum(self.vars_, self.reg)

  def _decision_function(self, X: np.ndarray) -> np.ndarray:
    """
    Return unnormalized log-posterior scores for each class:
      log p(y=c) + log p(x | y=c)
    Shape: (n_samples, n_classes)
    """
    if (
      self.classes_ is None
      or self.priors_ is None
      or self.means_ is None
      or self.vars_ is None
    ):
      raise RuntimeError("Call fit before inference.")
    diff = X[:, None, :] - self.means_[None, :, :]
    log_likelihood = -0.5 * (np.log(2.0 * np.pi * self.vars_)[None, :, :] + (diff ** 2) / self.vars_[None, :, :]).sum(axis=2)
    return log_likelihood + np.log(self.priors_)[None, :]

  def predict(self, X: np.ndarray) -> np.ndarray:
    if self.classes_ is None:
      raise RuntimeError("Call fit before predict.")
    scores = self._decision_function(X)
    return self.classes_[np.argmax(scores, axis=1)]

  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Return class posteriors with shape (n_samples, n_classes)."""
    scores = self._decision_function(X)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)
  