import numpy as np

from .base import Regression


class BayesianRegression(Regression):
    """Bayesian linear regression with optional evidence maximization.

    Parameters
    ----------
    sigma, tau
        Backward-compatible variance parameters. When provided, they override
        ``beta`` and ``alpha`` via ``beta = 1 / sigma^2`` and
        ``alpha = 1 / tau^2``.
    alpha, beta
        Prior precision and noise precision.
    penalize_bias
        When False, the bias term receives near-zero prior precision.
    max_evidence_iter, tol
        Controls evidence maximization updates.
    """

    def __init__(
        self,
        sigma: float | None = None,
        tau: float | None = None,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        penalize_bias: bool = False,
        max_evidence_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        if sigma is not None:
            if sigma <= 0:
                raise ValueError("sigma must be > 0")
            beta = 1.0 / float(sigma) ** 2
        if tau is not None:
            if tau <= 0:
                raise ValueError("tau must be > 0")
            alpha = 1.0 / float(tau) ** 2

        self.alpha = float(1.0 if alpha is None else alpha)
        self.beta = float(1.0 if beta is None else beta)
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be > 0.")

        self.penalize_bias = penalize_bias
        self.max_evidence_iter = max_evidence_iter
        self.tol = tol

        self.w_mean: np.ndarray | None = None
        self.w_cov: np.ndarray | None = None
        self.alpha_: float = self.alpha
        self.beta_: float = self.beta
        self.n_iter_: int = 0
        self.log_marginal_likelihood_history_: list[float] = []

    def _prior_precision_diag(self, d_aug: int) -> np.ndarray:
        diag = np.full(d_aug, self.alpha_, dtype=float)
        if not self.penalize_bias:
            diag[-1] = 1e-12
        return diag

    def _posterior(
        self,
        X_aug: np.ndarray,
        y: np.ndarray,
        *,
        alpha: float,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        d_aug = X_aug.shape[1]
        prior_diag = np.full(d_aug, alpha, dtype=float)
        if not self.penalize_bias:
            prior_diag[-1] = 1e-12

        precision = np.diag(prior_diag) + beta * (X_aug.T @ X_aug)
        cov = np.linalg.pinv(precision)
        mean = beta * cov @ X_aug.T @ y
        return mean.reshape(-1), cov

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        maximize_evidence: bool = False,
    ) -> None:
        X_aug = self._augment(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)

        self.alpha_ = self.alpha
        self.beta_ = self.beta
        self.log_marginal_likelihood_history_ = []
        self.n_iter_ = 1

        if maximize_evidence:
            self.maximize_evidence(X, y)
            return

        self.w_mean, self.w_cov = self._posterior(
            X_aug,
            y,
            alpha=self.alpha_,
            beta=self.beta_,
        )
        self.log_marginal_likelihood_history_.append(self.log_marginal_likelihood(X, y))

    def maximize_evidence(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        max_iter: int | None = None,
    ) -> None:
        X_aug = self._augment(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples = X_aug.shape[0]

        max_steps = self.max_evidence_iter if max_iter is None else max_iter
        alpha = float(self.alpha)
        beta = float(self.beta)
        self.log_marginal_likelihood_history_ = []

        for step in range(1, max_steps + 1):
            mean, cov = self._posterior(X_aug, y, alpha=alpha, beta=beta)

            if self.penalize_bias:
                penalized_mean = mean
                penalized_cov = cov
            else:
                penalized_mean = mean[:-1]
                penalized_cov = cov[:-1, :-1]

            n_penalized = penalized_mean.size
            gamma = float(n_penalized - alpha * np.trace(penalized_cov))
            denom_w = float(np.dot(penalized_mean, penalized_mean))
            resid = y - X_aug @ mean
            denom_e = float(np.dot(resid, resid))

            alpha_new = max(gamma / max(denom_w, 1e-12), 1e-12)
            beta_new = max((n_samples - gamma) / max(denom_e, 1e-12), 1e-12)

            self.alpha_ = alpha_new
            self.beta_ = beta_new
            self.w_mean = mean
            self.w_cov = cov
            self.log_marginal_likelihood_history_.append(self.log_marginal_likelihood(X, y))
            self.n_iter_ = step

            if abs(alpha_new - alpha) < self.tol and abs(beta_new - beta) < self.tol:
                break

            alpha, beta = alpha_new, beta_new

    def log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        X_aug = self._augment(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.w_cov is None or self.w_mean is None:
            mean, cov = self._posterior(X_aug, y, alpha=self.alpha_, beta=self.beta_)
        else:
            mean, cov = self.w_mean, self.w_cov

        d_aug = X_aug.shape[1]
        prior_diag = self._prior_precision_diag(d_aug)
        precision = np.diag(prior_diag) + self.beta_ * (X_aug.T @ X_aug)
        sign, logdet = np.linalg.slogdet(precision)
        if sign <= 0:
            raise RuntimeError("Posterior precision must be positive definite.")

        if self.penalize_bias:
            prior_term = self.alpha_ * np.dot(mean, mean)
        else:
            prior_term = self.alpha_ * np.dot(mean[:-1], mean[:-1])
        data_term = self.beta_ * np.dot(y - X_aug @ mean, y - X_aug @ mean)

        return float(
            0.5 * np.sum(np.log(np.maximum(prior_diag, 1e-12)))
            + 0.5 * len(y) * np.log(self.beta_)
            - 0.5 * data_term
            - 0.5 * prior_term
            - 0.5 * logdet
            - 0.5 * len(y) * np.log(2.0 * np.pi)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_mean is None:
            raise RuntimeError("Model not fitted")
        X_aug = self._augment(np.asarray(X, dtype=float))
        return X_aug @ self.w_mean

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.w_mean is None or self.w_cov is None:
            raise RuntimeError("Model not fitted")

        X_aug = self._augment(np.asarray(X, dtype=float))
        mean = X_aug @ self.w_mean
        model_var = np.einsum("ij,jk,ik->i", X_aug, self.w_cov, X_aug)
        noise_var = 1.0 / self.beta_
        total_var = noise_var + model_var
        std = np.sqrt(np.maximum(total_var, 0.0))
        return mean, std
