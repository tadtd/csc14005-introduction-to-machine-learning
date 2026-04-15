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

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature vs Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Regularization paths  (Ridge + Lasso side by side)
# ---------------------------------------------------------------------------

def plot_regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    alphas: np.ndarray | None = None,
    lasso_max_iter: int = 1_000,
    figsize: tuple = (18, 6),
) -> None:
    """Plot Ridge and Lasso coefficient paths as a function of log₁₀(α).

    Lasso uses warm start (coefficients from the previous α value are
    passed as the initial point for the next, in descending α order).

    Parameters
    ----------
    X, y : training data (should already be scaled).
    feature_names : list of feature name strings.
    alphas : 1-D array of regularization values.  Defaults to
             ``np.logspace(-4, 4, 80)``.
    lasso_max_iter : coordinate-descent passes per alpha value.
    """
    from models import RidgeRegression, LassoRegression   # local import to avoid circular deps

    if alphas is None:
        alphas = np.logspace(-4, 4, 80)

    alphas = np.asarray(alphas)

    # ----- Ridge path ----
    ridge_coefs = []
    for a in alphas:
        m = RidgeRegression(alpha=a)
        m.fit(X, y)
        ridge_coefs.append(m.coef_.copy())
    ridge_coefs = np.array(ridge_coefs)   # (n_alphas, d)

    # ----- Lasso path (warm start — descending α) -----
    n, d = X.shape
    lasso_coef   = np.zeros(d)
    lasso_intercept = float(np.mean(y))
    col_norms_sq = np.sum(X ** 2, axis=0) / n

    lasso_coefs_desc = []
    for a in alphas[::-1]:                # descend from large to small
        for _ in range(lasso_max_iter):
            max_delta = 0.0
            for j in range(d):
                r_j   = y - lasso_intercept - X @ lasso_coef + X[:, j] * lasso_coef[j]
                rho_j = float(X[:, j] @ r_j) / n
                z_j   = col_norms_sq[j]
                old   = lasso_coef[j]
                if z_j < 1e-12:
                    lasso_coef[j] = 0.0
                else:
                    sign = np.sign(rho_j)
                    lasso_coef[j] = float(sign * max(abs(rho_j) - a / 2.0, 0.0) / z_j)
                max_delta = max(max_delta, abs(lasso_coef[j] - old))
            lasso_intercept = float(np.mean(y - X @ lasso_coef))
            if max_delta < 1e-4:
                break
        lasso_coefs_desc.append(lasso_coef.copy())

    lasso_coefs = np.array(lasso_coefs_desc)[::-1]  # back to ascending α

    # ----- Plot -----
    log_alphas = np.log10(alphas)
    feature_names = list(feature_names)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for i, name in enumerate(feature_names):
        axes[0].plot(log_alphas, ridge_coefs[:, i], label=name)
    axes[0].set_xlabel("log₁₀(α)")
    axes[0].set_ylabel("Coefficient value")
    axes[0].set_title("Ridge — Regularization Path")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(alpha=0.3, linestyle="--")

    for i, name in enumerate(feature_names):
        axes[1].plot(log_alphas, lasso_coefs[:, i], label=name)
    axes[1].set_xlabel("log₁₀(α)")
    axes[1].set_ylabel("Coefficient value")
    axes[1].set_title("Lasso — Regularization Path")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(alpha=0.3, linestyle="--")

    plt.suptitle("Regularization Paths", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
