"""Utility functions for cross-validation, model selection, bias-variance
analysis, and visualisation for Part 1 (Regression).

All functions that the notebook imports are exported from this module:
    k_fold_cv, k_fold_cv_metrics, build_kfold_indices, kfold_rmse_per_model,
    bias_variance_bootstrap, forward_stepwise_selection, backward_elimination,
    paired_tests_against_best, plot_learning_curves, plot_residuals,
    plot_predicted_vs_actual, loss_curve, learning_curve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Sequence, Tuple, Any

from eval import mse as _mse, rmse as _rmse, mae as _mae, r2_score as _r2


# ===========================================================================
# K-Fold helpers
# ===========================================================================

def build_kfold_indices(
    n: int,
    k: int = 10,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return ``k`` (train_idx, val_idx) pairs for k-fold CV.

    Indices are shuffled before folding.

    Parameters
    ----------
    n : int  — total number of samples
    k : int  — number of folds
    random_state : int

    Returns
    -------
    List of length ``k``.  Each entry is (train_indices, val_indices).
    """
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1          # distribute remainder

    splits = []
    current = 0
    for fold_size in fold_sizes:
        val_idx   = indices[current : current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size :]])
        splits.append((train_idx, val_idx))
        current += fold_size

    return splits


def k_fold_cv(
    ModelClass,
    kwargs: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    random_state: int = 42,
) -> Tuple[float, float]:
    """K-fold cross-validation returning (mean_MSE, std_MSE).

    Parameters
    ----------
    ModelClass : class
        The model class.  An instance is created as ``ModelClass(**kwargs)``
        for each fold.
    kwargs : dict
        Constructor arguments forwarded to ``ModelClass``.
    X, y : arrays (should be scaled before calling).
    k : int — number of folds.

    Returns
    -------
    (mean_mse, std_mse)
    """
    splits = build_kfold_indices(len(y), k=k, random_state=random_state)
    fold_mse = []

    for train_idx, val_idx in splits:
        model = ModelClass(**kwargs)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        fold_mse.append(_mse(y_pred, y[val_idx]))

    return float(np.mean(fold_mse)), float(np.std(fold_mse))


def k_fold_cv_metrics(
    builder: Callable,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """K-fold CV returning mean ± std for MSE, RMSE, MAE, and R².

    Parameters
    ----------
    builder : zero-argument callable that returns a fresh model instance,
              e.g. ``lambda: LinearRegression(solver='normal')``.
              Compatible with sklearn estimators and Pipelines.
    X, y : arrays.
    k : int — number of folds.

    Returns
    -------
    dict with keys ``'mse'``, ``'rmse'``, ``'mae'``, ``'r2'``.
    Each value is ``(mean, std)`` across folds.
    """
    splits = build_kfold_indices(len(y), k=k, random_state=random_state)

    fold_metrics: Dict[str, List[float]] = {"mse": [], "rmse": [], "mae": [], "r2": []}

    for train_idx, val_idx in splits:
        model = builder()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        fold_metrics["mse"].append(_mse(y_pred, y[val_idx]))
        fold_metrics["rmse"].append(_rmse(y_pred, y[val_idx]))
        fold_metrics["mae"].append(_mae(y_pred, y[val_idx]))
        fold_metrics["r2"].append(_r2(y_pred, y[val_idx]))

    return {
        key: (float(np.mean(vals)), float(np.std(vals)))
        for key, vals in fold_metrics.items()
    }


def kfold_rmse_per_model(
    builder: Callable,
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Return a 1-D array of per-fold RMSE values for a given builder.

    Parameters
    ----------
    builder : zero-argument callable returning a fresh model instance.
    X, y : arrays.
    splits : pre-computed fold indices from ``build_kfold_indices``.

    Returns
    -------
    np.ndarray of shape ``(k,)``.
    """
    fold_rmse = []
    for train_idx, val_idx in splits:
        model = builder()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        fold_rmse.append(_rmse(y_pred, y[val_idx]))
    return np.array(fold_rmse)


# ===========================================================================
# Statistical tests
# ===========================================================================

def paired_tests_against_best(
    rmse_per_model: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired t-test and Wilcoxon signed-rank test vs. the best model.

    Parameters
    ----------
    rmse_per_model : dict mapping model name → 1-D array of per-fold RMSE.
    alpha : significance level.

    Returns
    -------
    pd.DataFrame with columns:
        Model, Mean RMSE, Std RMSE, vs_best (reference model name),
        t_pvalue, wilcoxon_pvalue, significant
    sorted by Mean RMSE ascending.
    """
    from scipy import stats as sp_stats

    means = {name: float(np.mean(v)) for name, v in rmse_per_model.items()}
    best_name = min(means, key=means.__getitem__)
    best_rmse = rmse_per_model[best_name]

    rows = []
    for name, fold_rmse in rmse_per_model.items():
        diff = fold_rmse - best_rmse

        if name == best_name or np.all(diff == 0):
            t_pval = 1.0
            w_pval = 1.0
        else:
            try:
                _, t_pval = sp_stats.ttest_rel(fold_rmse, best_rmse)
                t_pval = float(t_pval)
            except Exception:
                t_pval = float("nan")

            try:
                _, w_pval = sp_stats.wilcoxon(diff, alternative="two-sided")
                w_pval = float(w_pval)
            except Exception:
                # Wilcoxon requires at least one non-zero difference
                w_pval = float("nan")

        rows.append(
            {
                "Model":            name,
                "Mean RMSE":        round(means[name], 4),
                "Std RMSE":         round(float(np.std(fold_rmse)), 4),
                "vs_best":          best_name,
                "t_pvalue":         round(t_pval, 4),
                "wilcoxon_pvalue":  round(w_pval, 4) if not np.isnan(w_pval) else "n/a",
                "significant":      (t_pval < alpha) if not np.isnan(t_pval) else False,
            }
        )

    return pd.DataFrame(rows).sort_values("Mean RMSE").reset_index(drop=True)


# ===========================================================================
# Bias–Variance decomposition (bootstrap)
# ===========================================================================

def bias_variance_bootstrap(
    builder: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_boot: int = 200,
    random_state: int = 42,
) -> Dict[str, float]:
    """Estimate Bias², Variance, and Noise via bootstrap resampling.

    For each bootstrap sample b:
        1. Sample (with replacement) n training points.
        2. Fit ``builder()`` on the bootstrap sample.
        3. Predict on X_test → preds[b, :].

    Then:
        mean_pred[i]  = mean_b preds[b, i]
        bias2         = mean_i (mean_pred[i] - y_test[i])²
        variance      = mean_i var_b preds[b, i]
        total_mse     = mean_b mean_i (preds[b, i] - y_test[i])²
        noise         = max(total_mse - bias2 - variance, 0)

    Parameters
    ----------
    builder : zero-argument callable returning a fresh model instance.

    Returns
    -------
    dict with keys ``bias2``, ``variance``, ``noise``, ``mse``.
    """
    rng = np.random.RandomState(random_state)
    n_train = len(y_train)
    n_test  = len(y_test)

    all_preds = np.zeros((n_boot, n_test))

    for b in range(n_boot):
        idx = rng.choice(n_train, size=n_train, replace=True)
        model = builder()
        model.fit(X_train[idx], y_train[idx])
        all_preds[b] = model.predict(X_test)

    mean_pred  = all_preds.mean(axis=0)                          # (n_test,)
    bias2      = float(np.mean((mean_pred - y_test) ** 2))
    variance   = float(np.mean(all_preds.var(axis=0)))
    total_mse  = float(np.mean((all_preds - y_test[None, :]) ** 2))
    noise      = max(total_mse - bias2 - variance, 0.0)

    return {
        "bias2":    bias2,
        "variance": variance,
        "noise":    noise,
        "mse":      total_mse,
    }


# ===========================================================================
# Feature selection
# ===========================================================================

def forward_stepwise_selection(
    builder: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
) -> List[Dict]:
    """Forward stepwise feature selection using validation MSE as criterion.

    At each step, the feature whose addition most reduces val MSE is added.
    Stops when adding any remaining feature would increase val MSE.

    Parameters
    ----------
    builder : zero-argument callable returning a fresh model instance.

    Returns
    -------
    List of dicts, one per step:
        ``{'step', 'feature_added', 'selected', 'val_mse'}``
    """
    feature_names = list(feature_names)
    selected: List[int] = []
    remaining: List[int] = list(range(len(feature_names)))
    results = []

    best_val_mse = float("inf")

    while remaining:
        step_best_mse   = float("inf")
        step_best_feat  = -1

        for feat_idx in remaining:
            candidate = selected + [feat_idx]
            model = builder()
            model.fit(X_train[:, candidate], y_train)
            val_mse = _mse(model.predict(X_val[:, candidate]), y_val)

            if val_mse < step_best_mse:
                step_best_mse  = val_mse
                step_best_feat = feat_idx

        if step_best_mse >= best_val_mse:
            break   # no improvement — stop

        best_val_mse = step_best_mse
        selected.append(step_best_feat)
        remaining.remove(step_best_feat)

        results.append(
            {
                "step":          len(selected),
                "feature_added": feature_names[step_best_feat],
                "selected":      [feature_names[i] for i in selected],
                "val_mse":       round(step_best_mse, 4),
            }
        )

    return results


def backward_elimination(
    builder: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
) -> List[Dict]:
    """Backward feature elimination using validation MSE as criterion.

    At each step, the feature whose removal most reduces (or least increases)
    val MSE is eliminated.  Stops when no removal improves val MSE.

    Returns
    -------
    List of dicts, one per elimination step:
        ``{'step', 'feature_removed', 'remaining', 'val_mse'}``
    """
    feature_names = list(feature_names)
    remaining: List[int] = list(range(len(feature_names)))
    results = []

    # Baseline with all features
    model = builder()
    model.fit(X_train, y_train)
    best_val_mse = _mse(model.predict(X_val), y_val)

    step = 0
    while len(remaining) > 1:
        step_best_mse    = float("inf")
        step_best_remove = -1

        for feat_idx in remaining:
            candidate = [i for i in remaining if i != feat_idx]
            model = builder()
            model.fit(X_train[:, candidate], y_train)
            val_mse = _mse(model.predict(X_val[:, candidate]), y_val)

            if val_mse < step_best_mse:
                step_best_mse    = val_mse
                step_best_remove = feat_idx

        if step_best_mse >= best_val_mse:
            break   # no improvement — stop

        best_val_mse = step_best_mse
        remaining.remove(step_best_remove)
        step += 1

        results.append(
            {
                "step":             step,
                "feature_removed":  feature_names[step_best_remove],
                "remaining":        [feature_names[i] for i in remaining],
                "val_mse":          round(step_best_mse, 4),
            }
        )

    return results


# ===========================================================================
# Learning curves and loss curves
# ===========================================================================

def plot_learning_curves(
    ModelClass,
    kwargs: Dict[str, Any],
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    title: str = "Learning Curves",
    ScalerClass=None,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
) -> None:
    """Plot training and validation MSE as a function of training-set size.

    Parameters
    ----------
    ModelClass : the model class (not an instance).
    kwargs : constructor arguments forwarded to ``ModelClass``.
    X_train_raw, y_train : raw (unscaled) training data.
    X_val_raw, y_val : raw (unscaled) validation data.
    title : plot title.
    ScalerClass : optional scaler class (e.g. ``StandardScaler``).
        When provided, a fresh scaler is fitted on each training subset to
        avoid data leakage.
    train_sizes : fractions of the training set to evaluate at.
    """
    n_train = len(y_train)
    subset_ns = np.maximum((train_sizes * n_train).astype(int), 1)

    train_mse_list = []
    val_mse_list   = []

    for size in subset_ns:
        X_sub_raw = X_train_raw[:size]
        y_sub     = y_train[:size]

        if ScalerClass is not None:
            scaler  = ScalerClass()
            X_sub   = scaler.fit_transform(X_sub_raw)
            X_val_s = scaler.transform(X_val_raw)
        else:
            X_sub   = X_sub_raw
            X_val_s = X_val_raw

        model = ModelClass(**kwargs)
        model.fit(X_sub, y_sub)

        train_mse_list.append(_mse(model.predict(X_sub),   y_sub))
        val_mse_list.append(  _mse(model.predict(X_val_s), y_val))

    plt.figure(figsize=(8, 5))
    plt.plot(subset_ns, train_mse_list, "o-", label="Train MSE",      color="#1f77b4")
    plt.plot(subset_ns, val_mse_list,   "o-", label="Validation MSE", color="#ff7f0e")
    plt.xlabel("Training set size")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


def loss_curve(
    model,
    title: str = "Training Loss Curve",
    log_scale: bool = False,
) -> None:
    """Plot the per-iteration training loss stored in ``model.loss_history_``.

    Only applicable to gradient-descent models that populate
    ``loss_history_`` during ``fit``.

    Parameters
    ----------
    model : fitted model instance with a ``loss_history_`` attribute.
    title : plot title.
    log_scale : if True, use log scale on the y-axis.
    """
    if not hasattr(model, "loss_history_") or len(model.loss_history_) == 0:
        print("Model has no loss_history_.  Only GD-based models track this.")
        return

    iterations = np.arange(1, len(model.loss_history_) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(iterations, model.loss_history_, color="#1f77b4", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


def learning_curve(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
) -> None:
    """Learning curve for a pre-configured model instance.

    Re-fits the model on increasing fractions of ``X_train`` and records
    MSE on both train and validation sets.  Uses the model as provided
    (no scaler applied here — pass already-scaled data).

    Parameters
    ----------
    model : a model instance with ``.fit(X, y)`` and ``.predict(X)``.
    """
    n_train = len(y_train)
    subset_ns = np.maximum((train_sizes * n_train).astype(int), 1)

    train_mse_list, val_mse_list = [], []

    for size in subset_ns:
        model.fit(X_train[:size], y_train[:size])
        train_mse_list.append(_mse(model.predict(X_train[:size]), y_train[:size]))
        val_mse_list.append(  _mse(model.predict(X_val),          y_val))

    plt.figure(figsize=(8, 5))
    plt.plot(subset_ns, train_mse_list, "o-", label="Train MSE",      color="#1f77b4")
    plt.plot(subset_ns, val_mse_list,   "o-", label="Validation MSE", color="#ff7f0e")
    plt.xlabel("Training set size")
    plt.ylabel("MSE")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Standalone diagnostic plots (mirror of base class methods for direct use)
# ===========================================================================

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
) -> None:
    """Residual histogram (left) + Q-Q plot (right).

    Parameters
    ----------
    y_true, y_pred : arrays.
    title : optional label prepended to subplot titles.
    """
    import scipy.stats as sp_stats

    residuals = np.asarray(y_true, float) - np.asarray(y_pred, float)
    prefix = f"{title} — " if title else ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="#1f77b4")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{prefix}Residual Histogram")
    axes[0].grid(alpha=0.3, linestyle="--")

    sp_stats.probplot(residuals, plot=axes[1])
    axes[1].set_title(f"{prefix}Q-Q Plot")

    plt.tight_layout()
    plt.show()


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
) -> None:
    """Scatter of predicted values against actual values with the ideal line.

    Parameters
    ----------
    y_true, y_pred : arrays.
    title : plot title.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    r2  = _r2(y_pred, y_true)
    rmse_val = _rmse(y_pred, y_true)
    lo  = min(y_true.min(), y_pred.min())
    hi  = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, s=10, color="steelblue")
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title}\nR² = {r2:.4f}  RMSE = {rmse_val:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()
