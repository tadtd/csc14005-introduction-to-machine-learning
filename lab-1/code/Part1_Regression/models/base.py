import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict

from matplotlib import pyplot as plt
import scipy.stats as scipy_stats


class Regression(ABC):
    """Abstract base class for all regression models."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    # ------------------------------------------------------------------
    # Utility shared across solvers that augment X with a bias column
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(X: np.ndarray) -> np.ndarray:
        """Append a column of ones: (n, d) → (n, d+1)."""
        return np.c_[X, np.ones(X.shape[0])]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Fit the model to (X, y)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values for X."""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """Return MSE, RMSE, MAE, and R² as a dict.

        Argument order follows notebook convention: predictions first, then
        ground truth.
        """
        y_pred = np.asarray(y_pred, float)
        y_true = np.asarray(y_true, float)

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

        mse_val  = float(np.mean((y_pred - y_true) ** 2))
        rmse_val = float(np.sqrt(mse_val))
        mae_val  = float(np.mean(np.abs(y_pred - y_true)))
        r2_val   = 1.0 - ss_res / (ss_tot + 1e-12)

        return {
            "MSE":  mse_val,
            "RMSE": rmse_val,
            "MAE":  mae_val,
            "R2":   r2_val,
        }

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residuals",
    ) -> None:
        """Residual histogram (left) + Q-Q plot (right)."""
        residuals = np.asarray(y_true, float) - np.asarray(y_pred, float)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="#1f77b4")
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
        axes[0].set_xlabel("Residual")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"{title} — Residual Histogram")
        axes[0].grid(alpha=0.3, linestyle="--")

        scipy_stats.probplot(residuals, plot=axes[1])
        axes[1].set_title(f"{title} — Q-Q Plot")

        plt.tight_layout()
        plt.show()

    def plot_predicted_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predicted vs Actual",
    ) -> None:
        """Scatter plot of predicted against actual values with the ideal line."""
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)

        metrics = self.evaluate(y_pred, y_true)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.35, s=10, color="steelblue")
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{title}\nR² = {metrics['R2']:.4f}  RMSE = {metrics['RMSE']:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.show()
