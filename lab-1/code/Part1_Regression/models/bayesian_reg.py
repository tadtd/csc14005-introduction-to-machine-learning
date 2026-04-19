import numpy as np
from .base import Regression


class BayesianRegression(Regression):
  """
  Bayesian Linear Regression (explicit version)

  Model:
      y = X w + ε
      w ~ N(0, tau^2 I)
      ε ~ N(0, sigma^2)

  Equivalent precision form:
      alpha = 1 / tau^2   (prior precision)
      beta  = 1 / sigma^2 (noise precision)
  """

  def __init__(self, sigma: float = 1.0, tau: float = 1.0) -> None:
    if sigma <= 0:
      raise ValueError("sigma must be > 0")
    if tau <= 0:
      raise ValueError("tau must be > 0")

    # Convert variance → precision
    self.beta = 1.0 / sigma**2   # noise precision
    self.alpha = 1.0 / tau**2    # prior precision

    # Posterior parameters
    self.w_mean = None           # m_N
    self.w_cov = None            # S_N

  def fit(self, X: np.ndarray, y: np.ndarray) -> None:
    """
    Compute posterior p(w | D) = N(m_N, S_N)

    Formulas:
        S_N^{-1} = alpha I + beta X^T X
        S_N      = (S_N^{-1})^{-1}
        m_N      = beta S_N X^T y
    """

    # Add bias term
    X_aug = self._augment(X)   # shape (n, d+1)
    n_features = X_aug.shape[1]

    # ---- Step 1: Prior precision matrix ----
    # alpha * I
    prior_precision = self.alpha * np.eye(n_features)

    # Do not regularize bias (last column)
    prior_precision[-1, -1] = 1e-12

    # ---- Step 2: Posterior precision ----
    # S_N^{-1} = alpha I + beta X^T X
    posterior_precision = prior_precision + self.beta * (X_aug.T @ X_aug)

    # ---- Step 3: Posterior covariance ----
    # S_N = (S_N^{-1})^{-1}
    S_N = np.linalg.inv(posterior_precision)

    # ---- Step 4: Posterior mean ----
    # m_N = beta S_N X^T y
    m_N = self.beta * (S_N @ X_aug.T @ y)

    # Save results
    self.w_mean = m_N.reshape(-1)
    self.w_cov = S_N

  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Mean prediction:
        E[y*] = x*^T m_N
    """
    if self.w_mean is None:
      raise RuntimeError("Model not fitted")

    X_aug = self._augment(X)
    return X_aug @ self.w_mean

  def predict_with_uncertainty(self, X: np.ndarray):
    """
    Predictive distribution:

        p(y* | x*, D) = N(mean, variance)

    where:
        mean = x*^T m_N
        var  = sigma^2 + x*^T S_N x*
    """

    if self.w_mean is None or self.w_cov is None:
      raise RuntimeError("Model not fitted")

    X_aug = self._augment(X)

    # ---- Mean ----
    mean = X_aug @ self.w_mean

    # ---- Model uncertainty term ----
    # x*^T S_N x*
    model_var = np.einsum("ij,jk,ik->i", X_aug, self.w_cov, X_aug)

    # ---- Total variance ----
    # sigma^2 + model uncertainty
    noise_var = 1.0 / self.beta
    total_var = noise_var + model_var

    std = np.sqrt(np.maximum(total_var, 0.0))

    return mean, std