from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summarize_dataset(df: pd.DataFrame, target_col: str = "Class") -> pd.DataFrame:
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

	if target_col in df.columns:
		print("\nClass distribution:")
		print(df[target_col].value_counts())

	return missing


def summarize_numeric_statistics(
	X: pd.DataFrame,
	round_digits: int = 4,
) -> pd.DataFrame:
	"""Return descriptive statistics for numeric features.

	Includes count, mean, std, min, quartiles, median, max, IQR,
	and missing rate for each numeric column.
	"""
	numeric_df = X.select_dtypes(include=[np.number])
	if numeric_df.empty:
		print("No numeric columns found for descriptive statistics.")
		return pd.DataFrame()

	stats = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
	stats = stats.rename(
		columns={
			"25%": "q1",
			"50%": "median",
			"75%": "q3",
		}
	)
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

	stats = stats[ordered_cols].sort_values("std", ascending=False)
	return stats.round(round_digits)


def plot_class_distribution(
	y: Sequence,
	class_names: Sequence[str] | None = None,
	title: str = "Class Distribution",
) -> None:
	"""Plot class frequency as bar and pie charts."""
	y = np.asarray(y)
	labels, counts = np.unique(y, return_counts=True)

	if class_names is not None and len(class_names) == len(labels):
		tick_labels = [str(class_names[int(i)]) for i in labels]
	else:
		tick_labels = [str(i) for i in labels]

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))

	axes[0].bar(tick_labels, counts, color="steelblue", edgecolor="black", alpha=0.8)
	axes[0].set_title(title)
	axes[0].set_xlabel("Class")
	axes[0].set_ylabel("Count")
	axes[0].grid(axis="y", alpha=0.3, linestyle="--")

	axes[1].pie(counts, labels=tick_labels, autopct="%.1f%%", startangle=90)
	axes[1].set_title("Class Proportion")

	plt.tight_layout()
	plt.show()


def plot_correlation_heatmap(
	X: pd.DataFrame,
	title: str = "Feature Correlation Matrix",
	figsize: tuple[int, int] = (10, 8),
) -> None:
	"""Lower-triangle correlation heatmap for numeric features."""
	numeric_df = X.select_dtypes(include=[np.number])
	if numeric_df.empty:
		print("No numeric columns found for correlation heatmap.")
		return

	corr = numeric_df.corr()
	mask = np.triu(np.ones_like(corr, dtype=bool))

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
	plt.title(title)
	plt.tight_layout()
	plt.show()


def plot_top_feature_distributions(
	X: pd.DataFrame,
	y: Sequence,
	top_n: int = 6,
	max_rows: int = 2,
) -> None:
	"""Plot class-wise boxplots for top features by one-way ANOVA F-score proxy."""
	X_df = X.copy()
	y_arr = np.asarray(y)

	numeric_cols = list(X_df.select_dtypes(include=[np.number]).columns)
	if not numeric_cols:
		print("No numeric columns found for feature distribution plots.")
		return

	classes = np.unique(y_arr)
	scores = {}
	for col in numeric_cols:
		overall_mean = X_df[col].mean()
		between = 0.0
		within = 0.0

		for cls in classes:
			values = X_df.loc[y_arr == cls, col]
			n_c = len(values)
			if n_c == 0:
				continue
			cls_mean = values.mean()
			between += n_c * (cls_mean - overall_mean) ** 2
			within += ((values - cls_mean) ** 2).sum()

		scores[col] = between / (within + 1e-12)

	top_features = sorted(scores, key=scores.get, reverse=True)[:top_n]
	n_rows = min(max_rows, int(np.ceil(len(top_features) / 3)))
	n_cols = min(3, len(top_features))

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
	axes = np.array(axes).reshape(-1)

	plot_df = X_df[top_features].copy()
	plot_df["_target_"] = y_arr

	for i, col in enumerate(top_features):
		sns.boxplot(data=plot_df, x="_target_", y=col, ax=axes[i])
		axes[i].set_title(f"{col} by class")
		axes[i].set_xlabel("Class")
		axes[i].set_ylabel(col)
		axes[i].grid(alpha=0.2, linestyle="--")

	for j in range(len(top_features), len(axes)):
		axes[j].set_visible(False)

	plt.tight_layout()
	plt.show()
