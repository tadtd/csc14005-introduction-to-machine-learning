import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict

from matplotlib import pyplot as plt
import scipy.stats as scipy_stats
from eval import regression_report


class Regression(ABC):
    """Abstract base class for all regression models."""

    @staticmethod
    def _augment(X: np.ndarray) -> np.ndarray:
        """Append a column of ones: (n, d) → (n, d+1)."""
        return np.c_[X, np.ones(X.shape[0])]

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Fit the model to (X, y)."""
        raise NotImplementedError("fit method not implemented")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values for X."""
        raise NotImplementedError("predict method not implemented")

    def evaluate(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """Return MSE, RMSE, MAE, and R² as a dict.

        Argument order follows notebook convention: predictions first, then
        ground truth.
        """
        return regression_report(y_pred, y_true)

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
