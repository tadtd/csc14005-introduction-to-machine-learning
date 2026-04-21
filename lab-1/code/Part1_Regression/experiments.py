from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from eval import mse, rmse, mae, r2_score
from models import FeatureMapRegressor, LinearRegression
from basis import InteractionBasis


def evaluate_regressor(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    return {
        "mse": mse(y_pred, y_eval),
        "rmse": rmse(y_pred, y_eval),
        "mae": mae(y_pred, y_eval),
        "r2": r2_score(y_pred, y_eval),
    }


def validation_curve(
    builder_factory: Callable[[Any], Any],
    parameter_values: list[Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    parameter_name: str = "parameter",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in parameter_values:
        model = builder_factory(value)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        rows.append(
            {
                parameter_name: value,
                "train_mse": mse(train_pred, y_train),
                "val_mse": mse(val_pred, y_val),
                "train_rmse": rmse(train_pred, y_train),
                "val_rmse": rmse(val_pred, y_val),
            }
        )
    return pd.DataFrame(rows)


def basis_validation_curve(
    transformer_factory: Callable[[Any], Any],
    parameter_values: list[Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    base_model_factory: Callable[[], Any] | None = None,
    parameter_name: str = "basis_parameter",
) -> pd.DataFrame:
    if base_model_factory is None:
        base_model_factory = lambda: LinearRegression(solver="normal")

    def builder(value: Any) -> FeatureMapRegressor:
        return FeatureMapRegressor(
            transformer=transformer_factory(value),
            base_model=base_model_factory(),
        )

    return validation_curve(
        builder,
        parameter_values,
        X_train,
        y_train,
        X_val,
        y_val,
        parameter_name=parameter_name,
    )


def ablation_study(
    builder_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_groups: dict[str, list[int]],
) -> pd.DataFrame:
    """Remove one feature group at a time and measure performance drop."""
    baseline_model = builder_factory()
    baseline_metrics = evaluate_regressor(
        baseline_model,
        X_train,
        y_train,
        X_val,
        y_val,
    )
    n_features = X_train.shape[1]
    all_indices = np.arange(n_features)

    rows: list[dict[str, Any]] = [
        {
            "experiment": "baseline",
            "removed_group": "none",
            **baseline_metrics,
            "delta_val_mse": 0.0,
        }
    ]

    for group_name, group_indices in feature_groups.items():
        keep_indices = np.array([idx for idx in all_indices if idx not in set(group_indices)], dtype=int)
        model = builder_factory()
        metrics = evaluate_regressor(
            model,
            X_train[:, keep_indices],
            y_train,
            X_val[:, keep_indices],
            y_val,
        )
        rows.append(
            {
                "experiment": "feature_ablation",
                "removed_group": group_name,
                **metrics,
                "delta_val_mse": float(metrics["mse"] - baseline_metrics["mse"]),
            }
        )

    return pd.DataFrame(rows).sort_values("delta_val_mse", ascending=False).reset_index(drop=True)


def basis_ablation_study(
    basis_builders: dict[str, Callable[[], Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    base_model_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    if base_model_factory is None:
        base_model_factory = lambda: LinearRegression(solver="normal")

    rows: list[dict[str, Any]] = []
    for name, transformer_builder in basis_builders.items():
        model = FeatureMapRegressor(transformer_builder(), base_model_factory())
        metrics = evaluate_regressor(model, X_train, y_train, X_val, y_val)
        rows.append({"basis": name, **metrics})
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def interaction_effect_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    base_model_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    """Compare a plain linear model against one with interaction terms."""
    if base_model_factory is None:
        base_model_factory = lambda: LinearRegression(solver="normal")

    plain_model = base_model_factory()
    plain_metrics = evaluate_regressor(plain_model, X_train, y_train, X_val, y_val)

    interaction_model = FeatureMapRegressor(
        transformer=InteractionBasis(degree=2, include_bias=False),
        base_model=base_model_factory(),
    )
    interaction_metrics = evaluate_regressor(
        interaction_model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    return pd.DataFrame(
        [
            {"model": "plain_linear", **plain_metrics},
            {
                "model": "with_interactions",
                **interaction_metrics,
                "delta_mse_vs_plain": float(interaction_metrics["mse"] - plain_metrics["mse"]),
                "delta_rmse_vs_plain": float(interaction_metrics["rmse"] - plain_metrics["rmse"]),
            },
        ]
    )


def outlier_sensitivity_study(
    builder_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    contamination_levels: list[float],
    noise_scale: float = 5.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inject target outliers and compare robustness."""
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    baseline_model = builder_factory()
    baseline_metrics = evaluate_regressor(
        baseline_model, X_train, y_train, X_val, y_val
    )
    rows.append({"contamination": 0.0, **baseline_metrics})

    y_scale = float(np.std(y_train))
    for level in contamination_levels:
        y_noisy = np.asarray(y_train, dtype=float).copy()
        n_outliers = int(round(level * len(y_noisy)))
        if n_outliers > 0:
            indices = rng.choice(len(y_noisy), size=n_outliers, replace=False)
            noise = rng.normal(loc=0.0, scale=noise_scale * y_scale, size=n_outliers)
            y_noisy[indices] += noise

        model = builder_factory()
        metrics = evaluate_regressor(model, X_train, y_noisy, X_val, y_val)
        rows.append({"contamination": float(level), **metrics})

    return pd.DataFrame(rows).sort_values("contamination").reset_index(drop=True)
