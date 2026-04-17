import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Scalar regression metrics
# ---------------------------------------------------------------------------

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_pred, y_true)))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination R²."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def regression_report(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Return all four metrics as a dict."""
    return {
        "MSE":  mse(y_pred, y_true),
        "RMSE": rmse(y_pred, y_true),
        "MAE":  mae(y_pred, y_true),
        "R2":   r2_score(y_pred, y_true),
    }
