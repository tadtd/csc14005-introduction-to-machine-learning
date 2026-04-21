import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence


# ---------------------------------------------------------------------------
# Correlation and distribution
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (10, 8),
) -> None:
    """Lower-triangle annotated heatmap of all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns found.")
        return

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle

    n = len(corr.columns)
    figsize = (max(figsize[0], n * 0.7), max(figsize[1], n * 0.65))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Print high-level dataset diagnostics and return missing-value summary."""
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].rename("missing_count").to_frame()

    if missing.empty:
        print("\nMissing values: none")
    else:
        print("\nMissing values:")
        print(missing)

    return missing


def summarize_numeric_statistics(
    X: pd.DataFrame,
    round_digits: int = 4,
) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns in a table."""
    numeric_df = X.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns found for descriptive statistics.")
        return pd.DataFrame()

    stats = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats["missing_count"] = numeric_df.isna().sum()
    stats["missing_rate"] = numeric_df.isna().mean()

    ordered_cols = [
        "count",
        "mean",
        "std",
        "min",
        "q1",
        "median",
        "q3",
        "max",
        "iqr",
        "missing_count",
        "missing_rate",
    ]
    return stats[ordered_cols].sort_values("std", ascending=False).round(round_digits)


def plot_target_distribution(
    y: np.ndarray,
    target_name: str = "Target",
    bins: int = 50,
) -> None:
    """Histogram (left) + box plot (right) of the target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    axes[0].hist(y, bins=bins, edgecolor="black", alpha=0.7, color="#1f77b4")
    axes[0].set_title(f"Distribution of {target_name}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(target_name)
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].boxplot(
        y,
        patch_artist=True,
        boxprops=dict(facecolor="#1f77b4", alpha=0.6),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1].set_title(f"{target_name} — Box Plot")
    axes[1].set_ylabel(target_name)
    axes[1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()


def plot_feature_vs_target(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_cols: int = 4,
    alpha: float = 0.3,
    s: int = 8,
) -> None:
    """Grid of scatter plots: each feature against the target."""
    n_features = X.shape[1]
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(feature_names):
        axes[i].scatter(X[:, i], y, alpha=alpha, s=s, color="steelblue")
        axes[i].set_xlabel(name)
        axes[i].set_ylabel("Target")
        axes[i].set_title(name)
        axes[i].grid(alpha=0.3, linestyle="--")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature vs Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_feature_pair_scatter(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    sample_size: int = 1_500,
    random_state: int = 42,
    diag_kind: str = "hist",
) -> None:
    """Scatter-matrix for a small set of important variables.

    This complements the feature-vs-target plots by showing pairwise
    relationships among the most relevant variables requested in the project
    brief. The dataset is optionally down-sampled to keep the chart readable
    and the notebook responsive.
    """
    columns = [col for col in columns if col in df.columns]
    if len(columns) < 2:
        print("Need at least two valid columns to build a pair-scatter plot.")
        return

    pair_df = df[columns].copy()
    if sample_size is not None and sample_size > 0 and len(pair_df) > sample_size:
        pair_df = pair_df.sample(sample_size, random_state=random_state)

    grid = sns.pairplot(
        pair_df,
        corner=True,
        diag_kind=diag_kind,
        plot_kws={"alpha": 0.35, "s": 14, "edgecolor": "none"},
        diag_kws={"bins": 30, "edgecolor": "white"},
    )
    grid.fig.suptitle("Pairwise Scatter of Important Variables", y=1.02, fontsize=14, fontweight="bold")
    plt.show()
