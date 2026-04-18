import numpy as np

from .base import Regression


class RobustRegression(Regression):
	"""Robust regression implemented from scratch with NumPy only.

	Supported solvers
	-----------------
	method='huber':
		Iteratively Reweighted Least Squares (IRLS) minimizing Huber loss
		with optional L2 regularization on coefficients (bias is not penalized).
	method='ransac':
		Custom RANSAC loop that repeatedly samples minimal subsets, fits a
		linear model, scores inliers, and refits on the best consensus set.
	"""

	def __init__(
		self,
		method: str = "huber",
		epsilon: float = 1.35,
		alpha: float = 1e-4,
		max_iter: int = 1_000,
		min_samples: int | float | None = None,
		residual_threshold: float | None = None,
		random_state: int | None = None,
		tol: float = 1e-6,
	) -> None:
		self.method = method.lower()
		self.epsilon = epsilon
		self.alpha = alpha
		self.max_iter = max_iter
		self.min_samples = min_samples
		self.residual_threshold = residual_threshold
		self.random_state = random_state
		self.tol = tol

		self.theta_: np.ndarray | None = None
		self.inlier_mask_: np.ndarray | None = None
		self.loss_history_: list[float] = []
		self.n_iter_: int = 0

	@staticmethod
	def _solve_ols(X_b: np.ndarray, y: np.ndarray) -> np.ndarray:
		theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
		return np.asarray(theta, dtype=float).reshape(-1)

	@staticmethod
	def _solve_weighted(
		X_b: np.ndarray,
		y: np.ndarray,
		weights: np.ndarray,
		alpha: float,
	) -> np.ndarray:
		d_aug = X_b.shape[1]
		w = np.asarray(weights, dtype=float).reshape(-1)
		Xw = X_b * w[:, None]

		reg = alpha * np.eye(d_aug)
		reg[-1, -1] = 0.0

		A = X_b.T @ Xw + reg
		b = X_b.T @ (w * y)
		return np.linalg.solve(A, b)

	@staticmethod
	def _mad_scale(residuals: np.ndarray) -> float:
		med = float(np.median(residuals))
		mad = float(np.median(np.abs(residuals - med)))
		return max(1.4826 * mad, 1e-12)

	@staticmethod
	def _huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
		abs_r = np.abs(residuals)
		quad = 0.5 * residuals**2
		linear = delta * (abs_r - 0.5 * delta)
		return np.where(abs_r <= delta, quad, linear)

	def _fit_huber(self, X: np.ndarray, y: np.ndarray) -> None:
		X = np.asarray(X, dtype=float)
		y = np.asarray(y, dtype=float).reshape(-1)
		X_b = self._augment(X)
		theta = self._solve_ols(X_b, y)

		self.loss_history_ = []
		self.n_iter_ = 0

		for step in range(1, self.max_iter + 1):
			residuals = X_b @ theta - y
			sigma = self._mad_scale(residuals)
			delta = self.epsilon * sigma

			abs_r = np.abs(residuals)
			weights = np.ones_like(abs_r)
			outlier_mask = abs_r > delta
			weights[outlier_mask] = delta / abs_r[outlier_mask]

			theta_new = self._solve_weighted(X_b, y, weights, self.alpha)
			update_norm = float(np.linalg.norm(theta_new - theta))

			huber_term = float(np.mean(self._huber_loss(residuals, delta)))
			l2_term = self.alpha * float(np.sum(theta_new[:-1] ** 2))
			self.loss_history_.append(huber_term + l2_term)

			theta = theta_new
			self.n_iter_ = step

			if update_norm < self.tol:
				break

		self.theta_ = np.asarray(theta, dtype=float).reshape(-1)
		self.inlier_mask_ = None

	def _resolve_min_samples(self, n: int, d_aug: int) -> int:
		if self.min_samples is None:
			m = d_aug
		elif isinstance(self.min_samples, float):
			if not (0.0 < self.min_samples <= 1.0):
				raise ValueError("min_samples as float must be in (0, 1].")
			m = int(np.ceil(self.min_samples * n))
		else:
			m = int(self.min_samples)

		m = max(d_aug, m)
		m = min(n, m)
		if m < d_aug:
			raise ValueError("min_samples must be at least number of parameters.")
		return m

	def _fit_ransac(self, X: np.ndarray, y: np.ndarray) -> None:
		X_b = self._augment(X)
		n, d_aug = X_b.shape
		rng = np.random.default_rng(self.random_state)

		min_samples = self._resolve_min_samples(n, d_aug)

		if self.residual_threshold is None:
			theta0 = self._solve_ols(X_b, y)
			base_res = np.abs(X_b @ theta0 - y)
			threshold = 2.5 * self._mad_scale(base_res)
		else:
			threshold = float(self.residual_threshold)

		best_inlier_mask: np.ndarray | None = None
		best_inlier_count = -1
		best_mse = np.inf
		self.n_iter_ = 0

		for step in range(1, self.max_iter + 1):
			sample_idx = rng.choice(n, size=min_samples, replace=False)
			theta_candidate = self._solve_ols(X_b[sample_idx], y[sample_idx])

			residuals = np.abs(X_b @ theta_candidate - y)
			inlier_mask = residuals <= threshold
			inlier_count = int(np.sum(inlier_mask))

			if inlier_count >= d_aug:
				mse = float(np.mean((X_b[inlier_mask] @ theta_candidate - y[inlier_mask]) ** 2))
			else:
				mse = np.inf

			is_better = (inlier_count > best_inlier_count) or (
				inlier_count == best_inlier_count and mse < best_mse
			)
			if is_better:
				best_inlier_count = inlier_count
				best_mse = mse
				best_inlier_mask = inlier_mask.copy()

			self.n_iter_ = step

			if inlier_count == n:
				break

		if best_inlier_mask is None or best_inlier_count < d_aug:
			raise RuntimeError("RANSAC could not find a valid consensus set.")

		theta_final = self._solve_ols(X_b[best_inlier_mask], y[best_inlier_mask])
		self.theta_ = np.asarray(theta_final, dtype=float).reshape(-1)
		self.inlier_mask_ = best_inlier_mask
		self.loss_history_ = [best_mse]

	def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
		if self.method == "huber":
			self._fit_huber(X, y)
		elif self.method == "ransac":
			self._fit_ransac(X, y)
		else:
			raise ValueError(
				f"Unknown method {self.method!r}. Choose 'huber' or 'ransac'."
			)

	def predict(self, X: np.ndarray) -> np.ndarray:
		if self.theta_ is None:
			raise RuntimeError("Call fit before predict.")
		X = np.asarray(X, dtype=float)
		return self._augment(X) @ self.theta_
