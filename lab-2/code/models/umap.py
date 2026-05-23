import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cKDTree

from .base import BaseDR


class UMAP(BaseDR):
  """
  Uniform Manifold Approximation and Projection (UMAP).
  """

  def __init__(
    self,
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1,
    epochs: int = 200,
    negative_sample_rate: int = 5,
    learning_rate: float = 1.0,
    spread: float = 1.0,
    **kwargs,
  ):
    super().__init__(n_components=n_components, **kwargs)
    self.k = n_neighbors
    self.min_dist = min_dist
    self.epochs = epochs
    self.negative_sample_rate = negative_sample_rate
    self.learning_rate = learning_rate
    self.spread = spread
    self.embedding_ = None
    self._X_fit = None

  def _fit(self, X: np.ndarray) -> None:
    self.embedding_ = self._run_umap(X)
    self._X_fit = X

  def _fit_transform(self, X: np.ndarray) -> np.ndarray:
    self.embedding_ = self._run_umap(X)
    self._X_fit = X
    return self.embedding_

  def _transform(self, X: np.ndarray) -> np.ndarray:
    if X is self._X_fit:
      return self.embedding_
    if self._X_fit is not None and X.shape == self._X_fit.shape and np.allclose(X, self._X_fit):
      return self.embedding_
    raise NotImplementedError("Out-of-sample extension for UMAP is not implemented.")

  def _run_umap(self, X: np.ndarray) -> np.ndarray:
    N = X.shape[0]

    # phase 1: high-dimensional graph construction
    knn_indices, knn_distances = self._compute_nearest_neighbors(X, k=self.k)

    rho = np.zeros(N)
    sigma = np.zeros(N)
    for i in range(N):
      rho[i] = knn_distances[i, 0]
      sigma[i] = self._binary_search_sigma(knn_distances[i], rho[i], k=self.k)

    P_directed = np.zeros((N, N))
    for i in range(N):
      for j in knn_indices[i]:
        distance = np.linalg.norm(X[i] - X[j])
        P_directed[i, j] = np.exp(-max(0.0, distance - rho[i]) / sigma[i])

    P = P_directed + P_directed.T - (P_directed * P_directed.T)

    # phase 2: low-dimensional optimization
    Y = self._spectral_embedding(P, self.n_components)
    a, b = self._fit_phi_curve(self.min_dist, self.spread)
    Y = self._optimize_embedding(Y, P, a, b)

    return Y

  def _compute_nearest_neighbors(
    self, X: np.ndarray, k: int
  ) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(X)
    knn_distances, knn_indices = tree.query(X, k=k + 1, workers=-1) # workers=-1 to use all available cores
    return knn_indices[:, 1:], knn_distances[:, 1:]

  def _sigma_sum(self, knn_dists: np.ndarray, rho: float, sigma: float) -> float:
    if sigma <= 0:
      return np.inf
    shifted = np.maximum(0.0, knn_dists - rho)
    return np.sum(np.exp(-shifted / sigma))

  def _binary_search_sigma(
    self, knn_dists: np.ndarray, rho: float, k: int
  ) -> float:
    target = np.log2(k)
    lo, hi = 1e-5, 1.0

    while self._sigma_sum(knn_dists, rho, hi) < target and hi < 1e10:
      hi *= 2.0

    for _ in range(64):
      mid = (lo + hi) / 2.0
      if self._sigma_sum(knn_dists, rho, mid) < target:
        lo = mid
      else:
        hi = mid

    return (lo + hi) / 2.0

  def _spectral_embedding(self, P: np.ndarray, n_components: int) -> np.ndarray:
    D = np.diag(np.sum(P, axis=1))
    L = D - P
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    non_zero_indices = np.where(eigenvalues > 1e-5)[0]
    selected_indices = non_zero_indices[:n_components]
    Y = eigenvectors[:, selected_indices].astype(float)
    Y += self._rng.normal(0, 1e-4, Y.shape) # add small noise to break symmetry (standard UMAP practice)
    return Y

  def _fit_phi_curve(self, min_dist: float, spread: float) -> tuple[float, float]:
    if spread <= min_dist:
      raise ValueError(f"spread ({spread}) must be greater than min_dist ({min_dist})")

    def phi(d, a, b):
      return 1.0 / (1.0 + a * d ** (2 * b))

    xv = np.linspace(0.0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv <= min_dist] = 1.0
    mask = (xv > min_dist) & (xv < spread)
    yv[mask] = np.exp(-(xv[mask] - min_dist) / spread) # 

    pos = xv > 0
    a, b = curve_fit(phi, xv[pos], yv[pos], p0=(1.0, 1.0), maxfev=10000)[0]
    return float(a), float(b)

  def _positive_samples(self, P: np.ndarray) -> list[tuple[int, int, float]]:
    rows, cols = np.where(np.triu(P, k=1) > 0)
    weights = P[rows, cols]
    return list(zip(rows.tolist(), cols.tolist(), weights.tolist()))

  def _dist(self, y_i: np.ndarray, y_j: np.ndarray) -> float:
    return float(np.linalg.norm(y_i - y_j))

  def _compute_attractive_gradient(
    self,
    p_ij: float | None,
    q_ij: float,
    dist: float,
    a: float,
    b: float,
    y_i: np.ndarray,
    y_j: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    if p_ij is None:
      p_ij = 1.0
    if dist < 1e-8:
      return np.zeros_like(y_i), np.zeros_like(y_j)
    d2b = dist ** (2 * b)
    d2bm1 = dist ** (2 * b - 1)
    grad_coef = -2.0 * a * b * p_ij * d2bm1 / (1.0 + a * d2b)
    diff = (y_i - y_j) / dist
    grad = grad_coef * diff
    return grad, -grad

  def _compute_repulsive_gradient(
    self,
    q_ik: float,
    dist: float,
    a: float,
    b: float,
    y_i: np.ndarray,
    y_k: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    if dist < 1e-8:
      return np.zeros_like(y_i), np.zeros_like(y_k)
    d2b = dist ** (2 * b)
    dist_sq = dist * dist
    grad_coef = 2.0 * b / ((1e-3 + dist_sq) * (1.0 + a * d2b))
    diff = y_i - y_k
    grad = grad_coef * diff
    return grad, -grad

  def _optimize_embedding(
    self,
    Y: np.ndarray,
    P: np.ndarray,
    a: float,
    b: float,
  ) -> np.ndarray:
    Y = Y.copy()
    N = Y.shape[0]
    pairs = self._positive_samples(P)
    lr = self.learning_rate

    for epoch in range(self.epochs):
      for i, j, p_ij in pairs:
        if self._rng.random() > p_ij:
          continue
        dist_y = self._dist(Y[i], Y[j])
        q_ij = 1.0 / (1.0 + a * dist_y ** (2 * b))

        grad_i, grad_j = self._compute_attractive_gradient(1.0, q_ij, dist_y, a, b, Y[i], Y[j])
        grad_i = np.clip(grad_i, -4.0, 4.0)
        grad_j = np.clip(grad_j, -4.0, 4.0)
        Y[i] -= lr * grad_i
        Y[j] -= lr * grad_j

        for _ in range(self.negative_sample_rate):
          k = int(self._rng.integers(0, N))
          if k == i:
            continue
          dist_y_neg = self._dist(Y[i], Y[k])
          q_ik = 1.0 / (1.0 + a * dist_y_neg ** (2 * b))

          grad_i_neg, grad_k = self._compute_repulsive_gradient(q_ik, dist_y_neg, a, b, Y[i], Y[k])
          grad_i_neg = np.clip(grad_i_neg, -4.0, 4.0)
          grad_k = np.clip(grad_k, -4.0, 4.0)
          Y[i] -= lr * grad_i_neg
          Y[k] -= lr * grad_k

      lr *= 1.0 - (epoch + 1) / self.epochs

    return Y