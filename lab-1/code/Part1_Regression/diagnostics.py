from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def summarize_regression_dataset(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Return a compact data dictionary for the regression dataset."""
    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
                "is_target": bool(target_column is not None and column == target_column),
                "is_continuous": bool(pd.api.types.is_numeric_dtype(series)),
            }
        )
    return pd.DataFrame(rows)


def outliers_iqr(
    data: pd.DataFrame | np.ndarray,
    *,
    columns: Iterable[str] | None = None,
    whisker_width: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers per numeric feature using the IQR rule."""
    df = pd.DataFrame(data).copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    summary_rows: list[dict[str, float | int | str]] = []
    mask = pd.DataFrame(False, index=df.index, columns=list(columns))

    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - whisker_width * iqr
        upper = q3 + whisker_width * iqr
        col_mask = (values < lower) | (values > upper)
        mask[column] = col_mask.fillna(False)
        summary_rows.append(
            {
                "feature": column,
                "method": "IQR",
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "n_outliers": int(col_mask.sum()),
                "outlier_rate": float(col_mask.mean()),
            }
        )

    return pd.DataFrame(summary_rows), mask


def outliers_zscore(
    data: pd.DataFrame | np.ndarray,
    *,
    columns: Iterable[str] | None = None,
    threshold: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers per numeric feature using absolute z-score."""
    df = pd.DataFrame(data).copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    summary_rows: list[dict[str, float | int | str]] = []
    mask = pd.DataFrame(False, index=df.index, columns=list(columns))

    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std <= 1e-12:
            z = np.zeros(len(values), dtype=float)
        else:
            z = (values - mean) / std
        col_mask = np.abs(z) > threshold
        mask[column] = np.asarray(col_mask, dtype=bool)
        summary_rows.append(
            {
                "feature": column,
                "method": "Z-score",
                "threshold": float(threshold),
                "n_outliers": int(np.sum(col_mask)),
                "outlier_rate": float(np.mean(col_mask)),
            }
        )

    return pd.DataFrame(summary_rows), mask


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return pd.Series(
        {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "q1": float(np.quantile(residuals, 0.25)),
            "median": float(np.median(residuals)),
            "q3": float(np.quantile(residuals, 0.75)),
            "max": float(np.max(residuals)),
            "skew": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals)),
        }
    )


def breusch_pagan_test(
    X: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float]:
    """Breusch-Pagan heteroscedasticity test.

    The auxiliary regression is ``residuals^2 ~ [X, 1]``.
    """
    X = np.asarray(X, dtype=float)
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if X.shape[0] != residuals.shape[0]:
        raise ValueError("X and residuals must have the same number of rows.")

    z = residuals**2
    X_aux = np.c_[X, np.ones(X.shape[0])]
    coef, _, _, _ = np.linalg.lstsq(X_aux, z, rcond=None)
    z_hat = X_aux @ coef
    ss_res = float(np.sum((z - z_hat) ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)
    lm_stat = float(X.shape[0] * r_squared)
    dof = int(X.shape[1])
    p_value = float(stats.chi2.sf(lm_stat, df=dof))
    f_stat = float((r_squared / max(dof, 1)) / max((1.0 - r_squared) / max(X.shape[0] - dof - 1, 1), 1e-12))
    f_pvalue = float(stats.f.sf(f_stat, dof, max(X.shape[0] - dof - 1, 1)))
    return {
        "lm_stat": lm_stat,
        "lm_pvalue": p_value,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "r_squared_aux": float(r_squared),
        "dof": float(dof),
    }


def estimate_wls_weights(
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    *,
    n_bins: int = 10,
    min_variance: float = 1e-6,
) -> np.ndarray:
    """Estimate inverse-variance sample weights from residual dispersion."""
    fitted_values = np.asarray(fitted_values, dtype=float).reshape(-1)
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if fitted_values.shape != residuals.shape:
        raise ValueError("fitted_values and residuals must have the same shape.")

    order = np.argsort(fitted_values)
    bins = np.array_split(order, n_bins)
    weights = np.ones_like(fitted_values, dtype=float)

    for indices in bins:
        if len(indices) == 0:
            continue
        local_var = float(np.var(residuals[indices], ddof=1)) if len(indices) > 1 else float(np.var(residuals[indices]))
        local_var = max(local_var, min_variance)
        weights[indices] = 1.0 / local_var

    weights /= np.mean(weights)
    return weights


def heteroscedasticity_report(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    alpha: float = 0.05,
    n_bins: int = 10,
) -> dict[str, object]:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    bp = breusch_pagan_test(X, residuals)
    weights = estimate_wls_weights(y_pred, residuals, n_bins=n_bins)
    return {
        "residual_summary": residual_summary(y_true, y_pred),
        "breusch_pagan": bp,
        "suggest_wls": bool(bp["lm_pvalue"] < alpha),
        "estimated_weights": weights,
    }
