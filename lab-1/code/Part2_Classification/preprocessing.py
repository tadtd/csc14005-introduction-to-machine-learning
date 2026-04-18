from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load Dry Bean data from an Excel file and print basic info."""
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {file_path}") from e

    print(f"Loaded {file_path} -> shape {df.shape}")
    return df


@dataclass
class PreparedClassificationData:
    """Container for reusable train/val/test artifacts."""

    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: StandardScaler
    target_encoder: LabelEncoder
    feature_label_encoders: dict[str, LabelEncoder]
    numeric_cols: list[str]
    non_numeric_cols: list[str]


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into train/val/test while preserving the total val fraction."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_frac_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_frac_of_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _fit_transform_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    dict[str, LabelEncoder],
]:
    """Impute + encode categorical features with train-only fitting."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    non_numeric_cols = list(X_train.select_dtypes(exclude=[np.number]).columns)
    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="mean")
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = num_imputer.transform(X_val[numeric_cols])
        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    feature_label_encoders: dict[str, LabelEncoder] = {}
    if non_numeric_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[non_numeric_cols] = cat_imputer.fit_transform(X_train[non_numeric_cols])
        X_val[non_numeric_cols] = cat_imputer.transform(X_val[non_numeric_cols])
        X_test[non_numeric_cols] = cat_imputer.transform(X_test[non_numeric_cols])

        for col in non_numeric_cols:
            encoder = LabelEncoder()
            X_train[col] = encoder.fit_transform(X_train[col].astype(str))
            X_val[col] = encoder.transform(X_val[col].astype(str))
            X_test[col] = encoder.transform(X_test[col].astype(str))
            feature_label_encoders[col] = encoder

    return (
        X_train,
        X_val,
        X_test,
        numeric_cols,
        non_numeric_cols,
        feature_label_encoders,
    )


def prepare_classification_data(
    file_path: str,
    *,
    target_col: str = "Class",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    custom_df: Optional[pd.DataFrame] = None,
) -> PreparedClassificationData:
    """Run end-to-end preprocessing with train-only fitting to avoid leakage."""
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1:
        raise ValueError("test_size and val_size must be > 0 and sum to < 1.")

    df = custom_df.copy() if custom_df is not None else load_data(file_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw = _split_data(
        X,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    (
        X_train,
        X_val,
        X_test,
        numeric_cols,
        non_numeric_cols,
        feature_label_encoders,
    ) = _fit_transform_features(X_train, X_val, X_test)

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train_raw)
    y_val = target_encoder.transform(y_val_raw)
    y_test = target_encoder.transform(y_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(
        "Split sizes -> "
        f"train: {X_train_scaled.shape[0]}, "
        f"val: {X_val_scaled.shape[0]}, "
        f"test: {X_test_scaled.shape[0]}"
    )

    return PreparedClassificationData(
        df=df,
        X=X,
        y=y,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        target_encoder=target_encoder,
        feature_label_encoders=feature_label_encoders,
        numeric_cols=numeric_cols,
        non_numeric_cols=non_numeric_cols,
    )


