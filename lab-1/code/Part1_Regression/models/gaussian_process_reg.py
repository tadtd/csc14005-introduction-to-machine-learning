import numpy as np

from .base import Regression


class GaussianProcessRegression(Regression):
    """Gaussian Process Regression with a numerically stable RBF kernel.

    The implementation keeps the public API used by the notebook, but adds:
    - optional target normalization,
    - safe Cholesky factorization with adaptive jitter,
    - bounded log-space hyperparameter optimization,
    - backtracking line search to avoid NaN / overflow during tuning.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-2,
        jitter: float = 1e-10,
        normalize_y: bool = True,
    ) -> None:
        self.length_scale = float(length_scale)
        self.signal_variance = float(signal_variance)
        self.noise_variance = float(noise_variance)
        self.jitter = float(jitter)
        self.normalize_y = bool(normalize_y)
        if self.length_scale <= 0 or self.signal_variance <= 0 or self.noise_variance <= 0:
            raise ValueError("Kernel hyperparameters must be positive.")

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.L_: np.ndarray | None = None
        self.effective_jitter_: float | None = None
        self.log_marginal_likelihood_history_: list[float] = []
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0

    @staticmethod
    def _pairwise_sq_dist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = (
            np.sum(X1**2, axis=1, keepdims=True)
            + np.sum(X2**2, axis=1)
            - 2.0 * X1 @ X2.T
        )
        return np.maximum(sq_dist, 0.0)

    def _normalize_targets(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if not self.normalize_y:
            self.y_mean_ = 0.0
            self.y_scale_ = 1.0
            return y

        self.y_mean_ = float(np.mean(y))
        self.y_scale_ = float(np.std(y))
        if self.y_scale_ <= 1e-12:
            self.y_scale_ = 1.0
        return (y - self.y_mean_) / self.y_scale_

    def _transform_targets(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if not self.normalize_y:
            return y
        return (y - self.y_mean_) / self.y_scale_

    def _inverse_transform_targets(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if not self.normalize_y:
            return y
        return y * self.y_scale_ + self.y_mean_

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        sq_dist = self._pairwise_sq_dist(X1, X2)
        exponent = -sq_dist / (2.0 * self.length_scale * self.length_scale)
        exponent = np.clip(exponent, -700.0, 0.0)
        return self.signal_variance * np.exp(exponent)

    def _kernel_train(self, X: np.ndarray, jitter: float | None = None) -> np.ndarray:
        effective_jitter = self.jitter if jitter is None else float(jitter)
        K = self._rbf_kernel(X, X)
        K += (self.noise_variance + effective_jitter) * np.eye(X.shape[0], dtype=float)
        return K

    def _safe_cholesky(
        self,
        K: np.ndarray,
        *,
        initial_jitter: float | None = None,
        max_tries: int = 8,
    ) -> tuple[np.ndarray, float]:
        jitter = self.jitter if initial_jitter is None else float(initial_jitter)
        eye = np.eye(K.shape[0], dtype=float)
        base = np.asarray(K, dtype=float)
        for _ in range(max_tries):
            try:
                return np.linalg.cholesky(base + jitter * eye), jitter
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10.0, 1e-12)
        raise np.linalg.LinAlgError("Failed to compute a stable Cholesky factor.")

    def _posterior_from_normalized_targets(
        self,
        X: np.ndarray,
        y_norm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        K = self._kernel_train(X, jitter=0.0)
        L, used_jitter = self._safe_cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        return K, L, used_jitter, alpha

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        X = np.asarray(X, dtype=float)
        y_norm = self._normalize_targets(y)

        _, L, used_jitter, alpha = self._posterior_from_normalized_targets(X, y_norm)

        self.X_train_ = X
        self.y_train_ = y_norm
        self.L_ = L
        self.alpha_ = alpha
        self.effective_jitter_ = used_jitter
        self.log_marginal_likelihood_history_ = [self.log_marginal_likelihood(X, y)]

    def _solve_kernel_inverse(self, L: np.ndarray) -> np.ndarray:
        eye = np.eye(L.shape[0], dtype=float)
        return np.linalg.solve(L.T, np.linalg.solve(L, eye))

    def _log_marginal_likelihood_from_normalized_targets(
        self,
        X: np.ndarray,
        y_norm: np.ndarray,
    ) -> float:
        _, L, _, alpha = self._posterior_from_normalized_targets(X, y_norm)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return float(
            -0.5 * y_norm @ alpha
            - 0.5 * logdet
            - 0.5 * X.shape[0] * np.log(2.0 * np.pi)
        )

    def log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        X = np.asarray(X, dtype=float)
        y_norm = self._transform_targets(y)
        return self._log_marginal_likelihood_from_normalized_targets(X, y_norm)

    def _log_marginal_likelihood_gradient_from_normalized_targets(
        self,
        X: np.ndarray,
        y_norm: np.ndarray,
    ) -> dict[str, float]:
        X = np.asarray(X, dtype=float)
        sq_dist = self._pairwise_sq_dist(X, X)

        K_wo_noise = self._rbf_kernel(X, X)
        K = K_wo_noise + self.noise_variance * np.eye(X.shape[0], dtype=float)
        L, _ = self._safe_cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        K_inv = self._solve_kernel_inverse(L)
        common = np.outer(alpha, alpha) - K_inv

        dK_dlog_length = K_wo_noise * (sq_dist / (self.length_scale * self.length_scale))
        dK_dlog_signal = K_wo_noise
        dK_dlog_noise = self.noise_variance * np.eye(X.shape[0], dtype=float)

        grad_length = 0.5 * np.sum(common * dK_dlog_length)
        grad_signal = 0.5 * np.sum(common * dK_dlog_signal)
        grad_noise = 0.5 * np.sum(common * dK_dlog_noise)

        return {
            "log_length_scale": float(grad_length),
            "log_signal_variance": float(grad_signal),
            "log_noise_variance": float(grad_noise),
        }

    def log_marginal_likelihood_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        X = np.asarray(X, dtype=float)
        y_norm = self._transform_targets(y)
        return self._log_marginal_likelihood_gradient_from_normalized_targets(X, y_norm)

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> list[dict[str, float]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        y_norm = self._normalize_targets(y)

        params = np.log(
            np.array(
                [self.length_scale, self.signal_variance, self.noise_variance],
                dtype=float,
            )
        )
        lower_bounds = np.log(np.array([1e-3, 1e-6, 1e-8], dtype=float))
        upper_bounds = np.log(np.array([1e3, 1e6, 1e2], dtype=float))
        history: list[dict[str, float]] = []

        for step in range(max_iter):
            self.length_scale = float(np.exp(params[0]))
            self.signal_variance = float(np.exp(params[1]))
            self.noise_variance = float(np.exp(params[2]))

            current_lml = self._log_marginal_likelihood_from_normalized_targets(X, y_norm)
            grad = self._log_marginal_likelihood_gradient_from_normalized_targets(X, y_norm)
            grad_vec = np.array(
                [
                    grad["log_length_scale"],
                    grad["log_signal_variance"],
                    grad["log_noise_variance"],
                ],
                dtype=float,
            )
            grad_vec = np.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = float(np.linalg.norm(grad_vec))

            history.append(
                {
                    "iteration": float(step),
                    "log_marginal_likelihood": float(current_lml),
                    "length_scale": self.length_scale,
                    "signal_variance": self.signal_variance,
                    "noise_variance": self.noise_variance,
                    "grad_norm": grad_norm,
                }
            )

            if grad_norm < tol:
                break

            direction = grad_vec / max(grad_norm, 1.0)
            step_size = float(learning_rate)
            accepted = False

            for _ in range(12):
                candidate = np.clip(params + step_size * direction, lower_bounds, upper_bounds)
                self.length_scale = float(np.exp(candidate[0]))
                self.signal_variance = float(np.exp(candidate[1]))
                self.noise_variance = float(np.exp(candidate[2]))
                candidate_lml = self._log_marginal_likelihood_from_normalized_targets(X, y_norm)
                if np.isfinite(candidate_lml) and candidate_lml >= current_lml:
                    params = candidate
                    accepted = True
                    break
                step_size *= 0.5

            if not accepted or step_size < tol:
                break

        self.length_scale = float(np.exp(params[0]))
        self.signal_variance = float(np.exp(params[1]))
        self.noise_variance = float(np.exp(params[2]))
        self.fit(X, y)
        self.log_marginal_likelihood_history_ = [
            row["log_marginal_likelihood"] for row in history
        ]
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.alpha_ is None:
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        K_star = self._rbf_kernel(X, self.X_train_)
        mean_norm = K_star @ self.alpha_
        return self._inverse_transform_targets(mean_norm)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train_ is None or self.alpha_ is None or self.L_ is None:
            raise RuntimeError("Call fit before predict_with_uncertainty.")

        X = np.asarray(X, dtype=float)
        K_star = self._rbf_kernel(X, self.X_train_)
        mean_norm = K_star @ self.alpha_

        v = np.linalg.solve(self.L_, K_star.T)
        K_xx = self._rbf_kernel(X, X) + self.jitter * np.eye(X.shape[0], dtype=float)
        # Return predictive uncertainty for the observed target y* by default,
        # so the observation noise term is included in the variance.
        var_norm = np.diag(K_xx) - np.sum(v * v, axis=0) + self.noise_variance
        var_norm = np.maximum(var_norm, 0.0)
        std_norm = np.sqrt(var_norm)

        mean = self._inverse_transform_targets(mean_norm)
        std = std_norm * self.y_scale_
        return mean, std
