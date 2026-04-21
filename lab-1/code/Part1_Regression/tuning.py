from __future__ import annotations

from itertools import product
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from eval import mae, mse, r2_score, rmse
from models import ElasticNetRegression, KernelRidgeRegression, LassoRegression, RidgeRegression
from utils import build_kfold_indices


RegressionBuilder = Callable[[dict[str, Any]], Any]


def _iter_param_grid(param_grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [list(param_grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _score_name_to_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    metric = metric.lower()
    if metric == "mse":
        return mse(y_pred, y_true)
    if metric == "rmse":
        return rmse(y_pred, y_true)
    if metric == "mae":
        return mae(y_pred, y_true)
    if metric == "r2":
        return r2_score(y_pred, y_true)
    raise ValueError("metric must be one of {'mse', 'rmse', 'mae', 'r2'}.")


def grid_search_cv(
    builder: RegressionBuilder,
    param_grid: dict[str, Iterable[Any]],
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 10,
    random_state: int = 42,
    metric: str = "mse",
    greater_is_better: bool = False,
    ScalerClass=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a parameter grid with k-fold CV."""
    splits = build_kfold_indices(len(y), k=k, random_state=random_state)
    rows: list[dict[str, Any]] = []

    for params in _iter_param_grid(param_grid):
        fold_scores: list[float] = []
        fit_times: list[float] = []
        for train_idx, val_idx in splits:
            model = builder(params)
            X_train_fold = np.asarray(X[train_idx])
            X_val_fold = np.asarray(X[val_idx])
            if ScalerClass is not None:
                scaler = ScalerClass()
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_val_fold = scaler.transform(X_val_fold)
            start = perf_counter()
            model.fit(X_train_fold, y[train_idx])
            fit_times.append(perf_counter() - start)
            y_pred = model.predict(X_val_fold)
            fold_scores.append(_score_name_to_value(y[val_idx], y_pred, metric))

        row = dict(params)
        row.update(
            {
                f"mean_{metric}": float(np.mean(fold_scores)),
                f"std_{metric}": float(np.std(fold_scores)),
                "mean_fit_time": float(np.mean(fit_times)),
            }
        )
        rows.append(row)

    results = pd.DataFrame(rows)
    ascending = not greater_is_better
    results = results.sort_values(f"mean_{metric}", ascending=ascending).reset_index(drop=True)
    best_params = results.iloc[0].to_dict()
    return results, best_params


def ridge_regularization_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Iterable[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        model = RidgeRegression(alpha=float(alpha))
        model.fit(X_train, y_train)
        row: dict[str, Any] = {
            "alpha": float(alpha),
            "log_alpha": float(np.log10(alpha)),
            "val_mse": mse(model.predict(X_val), y_val),
            "val_rmse": rmse(model.predict(X_val), y_val),
        }
        for idx, coef in enumerate(model.coef_):
            row[f"coef_{idx}"] = float(coef)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def lasso_regularization_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Iterable[float],
    *,
    warm_start: bool = True,
    max_iter: int = 5_000,
    tol: float = 1e-4,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_theta: np.ndarray | None = None
    for alpha in sorted(float(a) for a in alphas):
        model = LassoRegression(alpha=alpha, max_iter=max_iter, tol=tol)
        fit_kwargs: dict[str, Any] = {}
        if warm_start and previous_theta is not None:
            fit_kwargs["initial_theta"] = previous_theta
        model.fit(X_train, y_train, **fit_kwargs)
        previous_theta = model.theta_.copy()

        row: dict[str, Any] = {
            "alpha": alpha,
            "log_alpha": float(np.log10(alpha)),
            "val_mse": mse(model.predict(X_val), y_val),
            "val_rmse": rmse(model.predict(X_val), y_val),
            "n_nonzero": int(np.count_nonzero(np.abs(model.coef_) > 1e-10)),
            "n_iter": int(model.n_iter_),
            "warm_start": bool(warm_start),
        }
        for idx, coef in enumerate(model.coef_):
            row[f"coef_{idx}"] = float(coef)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def elastic_net_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lambda1_values: Iterable[float],
    lambda2_values: Iterable[float],
    k: int = 10,
    random_state: int = 42,
    max_iter: int = 1_000,
    tol: float = 1e-4,
    ScalerClass=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return grid_search_cv(
        lambda params: ElasticNetRegression(
            lambda1=float(params["lambda1"]),
            lambda2=float(params["lambda2"]),
            max_iter=max_iter,
            tol=tol,
        ),
        {"lambda1": list(lambda1_values), "lambda2": list(lambda2_values)},
        X,
        y,
        k=k,
        random_state=random_state,
        metric="mse",
        ScalerClass=ScalerClass,
    )


def kernel_ridge_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alphas: Iterable[float],
    gammas: Iterable[float],
    kernels: Iterable[str] = ("rbf", "poly"),
    degrees: Iterable[int] = (2, 3),
    coef0_values: Iterable[float] = (1.0,),
    k: int = 10,
    random_state: int = 42,
    ScalerClass=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return grid_search_cv(
        lambda params: KernelRidgeRegression(
            alpha=float(params["alpha"]),
            gamma=float(params["gamma"]),
            kernel=str(params["kernel"]),
            degree=int(params["degree"]),
            coef0=float(params["coef0"]),
        ),
        {
            "alpha": list(alphas),
            "gamma": list(gammas),
            "kernel": list(kernels),
            "degree": list(degrees),
            "coef0": list(coef0_values),
        },
        X,
        y,
        k=k,
        random_state=random_state,
        metric="mse",
        ScalerClass=ScalerClass,
    )


def lasso_warm_start_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Iterable[float],
    *,
    k: int = 10,
    random_state: int = 42,
    max_iter: int = 5_000,
    tol: float = 1e-4,
    metric: str = "mse",
    ScalerClass=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cross-validate Lasso along a warm-start regularization path.

    Each fold traverses the alphas from large to small so that the solution at
    the previous alpha is reused as the initialization for the next alpha.
    """
    alphas_sorted = sorted(float(alpha) for alpha in alphas)
    alphas_desc = list(reversed(alphas_sorted))
    splits = build_kfold_indices(len(y), k=k, random_state=random_state)

    fold_scores_by_alpha: dict[float, list[float]] = {alpha: [] for alpha in alphas_sorted}
    fold_iters_by_alpha: dict[float, list[int]] = {alpha: [] for alpha in alphas_sorted}

    for train_idx, val_idx in splits:
        X_train_fold = np.asarray(X[train_idx])
        X_val_fold = np.asarray(X[val_idx])
        if ScalerClass is not None:
            scaler = ScalerClass()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_val_fold = scaler.transform(X_val_fold)

        initial_theta: np.ndarray | None = None
        for alpha in alphas_desc:
            model = LassoRegression(alpha=alpha, max_iter=max_iter, tol=tol)
            fit_kwargs: dict[str, Any] = {}
            if initial_theta is not None:
                fit_kwargs["initial_theta"] = initial_theta
            model.fit(X_train_fold, y[train_idx], **fit_kwargs)
            y_pred = model.predict(X_val_fold)
            score = _score_name_to_value(y[val_idx], y_pred, metric)
            fold_scores_by_alpha[alpha].append(score)
            fold_iters_by_alpha[alpha].append(int(model.n_iter_))
            initial_theta = model.theta_.copy()

    rows: list[dict[str, Any]] = []
    for alpha in alphas_sorted:
        rows.append(
            {
                "alpha": alpha,
                "log_alpha": float(np.log10(alpha)),
                f"mean_{metric}": float(np.mean(fold_scores_by_alpha[alpha])),
                f"std_{metric}": float(np.std(fold_scores_by_alpha[alpha])),
                "mean_n_iter": float(np.mean(fold_iters_by_alpha[alpha])),
                "warm_start_cv": True,
            }
        )

    results = pd.DataFrame(rows).sort_values(f"mean_{metric}").reset_index(drop=True)
    best_params = results.iloc[0].to_dict()
    return results, best_params


def benchmark_linear_solvers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    solver_configs: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    """Compare solver speed, convergence and validation error."""
    rows: list[dict[str, Any]] = []
    for config in solver_configs:
        params = dict(config)
        model_class = params.pop("model_class")
        label = params.pop("label", model_class.__name__)
        model = model_class(**params)
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_time = perf_counter() - start
        y_pred = model.predict(X_val)
        loss_history = getattr(model, "loss_history_", None)
        if loss_history is None or len(loss_history) == 0:
            final_loss = mse(model.predict(X_train), y_train)
        else:
            final_loss = float(loss_history[-1])
        rows.append(
            {
                "model": label,
                "fit_time_seconds": float(getattr(model, "fit_time_seconds_", fit_time)),
                "n_iter": int(getattr(model, "n_iter_", 1)),
                "final_loss": float(final_loss),
                "val_mse": mse(y_pred, y_val),
                "val_rmse": rmse(y_pred, y_val),
                "learning_rate_schedule": getattr(model, "learning_rate_schedule", "n/a"),
                "solver": getattr(model, "solver", "n/a"),
            }
        )
    return pd.DataFrame(rows).sort_values(["val_mse", "fit_time_seconds"]).reset_index(drop=True)
