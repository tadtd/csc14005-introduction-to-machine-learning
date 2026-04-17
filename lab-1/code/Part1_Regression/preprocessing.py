import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as _train_test_split
from typing import Tuple


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


def impute_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill missing values in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (should be the training set only to avoid leakage).
    strategy : {'median', 'mean', 'zero'}

    Returns
    -------
    pd.DataFrame with NaNs filled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if strategy == "median":
        fill_values = df[numeric_cols].median()
    elif strategy == "mean":
        fill_values = df[numeric_cols].mean()
    elif strategy == "zero":
        fill_values = 0.0
    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose 'median', 'mean', or 'zero'.")

    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing > 0:
        df[numeric_cols] = df[numeric_cols].fillna(fill_values)
        print(f"Imputed {n_missing} missing value(s) using '{strategy}' strategy.")
    else:
        print("No missing values found.")

    return df


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train / validation / test sets.

    Strategy:
        1. Hold out ``test_size`` fraction as test set.
        2. Hold out ``val_size / (1 - test_size)`` of the remainder as
           validation so that the validation fraction of the *total* dataset
           equals ``val_size``.

    Parameters
    ----------
    X, y : np.ndarray
    val_size : float  (fraction of total)
    test_size : float (fraction of total)
    random_state : int

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First, split off the test set
    X_temp, X_test, y_temp, y_test = _train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Then split the remainder into train + validation
    val_frac_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = _train_test_split(
        X_temp, y_temp, test_size=val_frac_of_temp, random_state=random_state
    )

    print(
        f"Split sizes -> train: {len(X_train)}  "
        f"val: {len(X_val)}  test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


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
