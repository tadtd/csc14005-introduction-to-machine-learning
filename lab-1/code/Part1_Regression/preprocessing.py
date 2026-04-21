from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset and print a short summary.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {file_path}  →  shape {df.shape}")
    return df


@dataclass
class PreparedRegressionData:
    """Container for regression preprocessing artifacts."""

    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    X_train_raw: pd.DataFrame
    X_val_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    scaler: StandardScaler
    numeric_cols: list[str]
    drop_cols: list[str]


def _fit_and_apply_numeric_imputer(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Impute numeric columns using statistics learned from the training split."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        return X_train, X_val, X_test

    if strategy == "median":
        fill_values = X_train[numeric_cols].median()
    elif strategy == "mean":
        fill_values = X_train[numeric_cols].mean()
    elif strategy == "zero":
        fill_values = 0.0
    else:
        raise ValueError("Unknown strategy {!r}. Choose 'median', 'mean', or 'zero'.".format(strategy))

    missing_count = int(X_train[numeric_cols].isna().sum().sum() + X_val[numeric_cols].isna().sum().sum() + X_test[numeric_cols].isna().sum().sum())
    if missing_count > 0:
        X_train[numeric_cols] = X_train[numeric_cols].fillna(fill_values)
        X_val[numeric_cols] = X_val[numeric_cols].fillna(fill_values)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(fill_values)
        print(f"Imputed {missing_count} missing value(s) using '{strategy}' strategy on train statistics.")
    else:
        print("No missing values found.")

    return X_train, X_val, X_test


def scale_features(
    X_train: np.ndarray,
    *others: np.ndarray,
) -> Tuple:
    """Fit a StandardScaler on X_train and transform all provided arrays.

    Parameters
    ----------
    X_train : np.ndarray
        Training features used to fit the scaler.
    *others : np.ndarray
        Additional arrays to transform (e.g. X_val, X_test).

    Returns
    -------
    (scaler, X_train_scaled, *others_scaled)

    Example
    -------
    scaler, X_tr, X_val, X_te = scale_features(X_train, X_val, X_test)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    others_scaled = tuple(scaler.transform(arr) for arr in others)
    return (scaler, X_train_scaled) + others_scaled


def prepare_regression_data(
    file_path: str,
    *,
    target_col: str = "cnt",
    drop_cols: tuple[str, ...] = ("timestamp",),
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
    impute_strategy: str = "median",
    custom_df: Optional[pd.DataFrame] = None,
) -> PreparedRegressionData:
    """Load, clean, split, and scale regression data with train-only fitting.

    The train/validation/test fractions are computed on the total dataset.
    Numeric missing values are imputed using statistics from the training split only.
    """
    df = custom_df.copy() if custom_df is not None else load_data(file_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in input data.")

    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    drop_cols_existing = [col for col in drop_cols if col in X.columns]
    feature_names = [col for col in X.columns if col not in drop_cols_existing]
    X = X[feature_names]

    X_temp, X_test_raw, y_temp, y_test = _train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    val_frac_of_temp = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = _train_test_split(
        X_temp,
        y_temp,
        test_size=val_frac_of_temp,
        random_state=random_state,
    )

    X_train_raw, X_val_raw, X_test_raw = _fit_and_apply_numeric_imputer(
        X_train_raw,
        X_val_raw,
        X_test_raw,
        strategy=impute_strategy,
    )

    scaler, X_train, X_val, X_test = scale_features(X_train_raw.values, X_val_raw.values, X_test_raw.values)

    print(
        f"Split sizes -> train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}"
    )

    numeric_cols = list(X_train_raw.select_dtypes(include=[np.number]).columns)

    return PreparedRegressionData(
        df=df,
        X=X,
        y=y,
        feature_names=feature_names,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
        y_test=np.asarray(y_test),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        scaler=scaler,
        numeric_cols=numeric_cols,
        drop_cols=drop_cols_existing,
    )
