from copy import deepcopy
from collections.abc import Mapping

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2, binomtest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from typing import Dict, Any, Tuple, Optional, Callable, Literal


def kfold_cross_val_score(
  model: Any,
  X: np.ndarray,
  y: np.ndarray,
  k: int = 5,
  average: Literal["micro", "macro", "weighted", "binary"] = "weighted",
  *,
  random_state: int = 42,
  shuffle: bool = True,
  model_builder: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
  """Run stratified k-fold CV and return per-fold metrics with mean ± std.

  Parameters
  ----------
  model
    A model implementing ``fit(X, y)``, ``predict(X)``, and ``evaluate(y_pred, y_true, average=...)``.
  X, y
    Feature matrix and labels.
  k
    Number of folds. Defaults to 5.
  average
    Aggregation mode passed to ``model.evaluate``.
  random_state, shuffle
    Controls fold shuffling for reproducible splits.
  model_builder
    Optional factory to construct a fresh model per fold. If omitted, ``deepcopy(model)`` is used.
  """
  X = np.asarray(X)
  y = np.asarray(y)

  if X.shape[0] != y.shape[0]:
    raise ValueError(f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}.")
  if k < 2:
    raise ValueError(f"k must be at least 2, got {k}.")
  if k > y.shape[0]:
    raise ValueError(f"k must be <= number of samples ({y.shape[0]}), got {k}.")

  splitter = StratifiedKFold(
    n_splits=k,
    shuffle=shuffle,
    random_state=random_state if shuffle else None,
  )

  metric_names = ("accuracy", "precision", "recall", "f1-score")
  fold_values = {name: [] for name in metric_names}
  per_fold: list[Dict[str, float]] = []

  for fold_id, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
    fold_model = model_builder() if model_builder is not None else deepcopy(model)
    fold_model.fit(X[train_idx], y[train_idx])

    y_val_pred = fold_model.predict(X[val_idx])
    metrics = fold_model.evaluate(y_val_pred, y[val_idx], average=average)

    row: Dict[str, float] = {"fold": float(fold_id)}
    for name in metric_names:
      score = float(metrics[name])
      fold_values[name].append(score)
      row[name] = score
    per_fold.append(row)

  summary: Dict[str, Dict[str, float | str]] = {}
  for name in metric_names:
    vals = np.asarray(fold_values[name], dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    summary[name] = {
      "mean": mean,
      "std": std,
      "mean_std": f"{mean:.4f} +- {std:.4f}",
    }

  return {
    "k": int(k),
    "average": average,
    "metrics_per_fold": per_fold,
    "summary": summary,
  }


def _precision_recall_inputs(
  model: Any,
  X: np.ndarray,
  y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
  """Return the target representation, score array, and binary/multiclass flag."""
  if not hasattr(model, "predict_proba"):
    raise AttributeError("Model must implement predict_proba(X).")

  y_true = np.asarray(y_true)
  y_score = np.asarray(model.predict_proba(X))
  classes = np.asarray(getattr(model, "classes_", np.unique(y_true)))

  if classes.size < 2:
    raise ValueError("Precision-recall curves require at least two distinct classes.")

  if y_score.ndim == 1:
    if classes.size != 2:
      raise ValueError("1D probability scores are only supported for binary classification.")
    y_bin = (y_true == classes[1]).astype(int)
    return y_bin, y_score, True

  if y_score.ndim != 2:
    raise ValueError("predict_proba must return a 1D or 2D array of probabilities.")

  if classes.size == 2:
    if y_score.shape[1] != 2:
      raise ValueError("Binary classifiers must return probabilities with shape (n_samples, 2).")
    y_bin = (y_true == classes[1]).astype(int)
    return y_bin, y_score[:, 1], True

  if y_score.shape[1] != classes.size:
    raise ValueError(
      "Multiclass probabilities must have one column per class in model.classes_."
    )

  y_bin = label_binarize(y_true, classes=classes)
  if y_bin.shape[1] != classes.size:
    raise RuntimeError("Unexpected label binarization shape for multiclass targets.")

  return y_bin, y_score, False


def plot_precision_recall_curve(
  model: Any,
  X: np.ndarray,
  y_true: np.ndarray,
  title: str = "Precision-Recall Curve",
) -> float:
  """Plot a precision-recall curve for a fitted model and return its AP."""
  y_target, y_score, is_binary = _precision_recall_inputs(model, X, y_true)

  if is_binary:
    precision, recall, _ = precision_recall_curve(y_target, y_score)
    average_precision = float(average_precision_score(y_target, y_score))
  else:
    precision, recall, _ = precision_recall_curve(y_target.ravel(), y_score.ravel())
    average_precision = float(average_precision_score(y_target, y_score, average="micro"))

  plt.figure(figsize=(8, 6))
  plt.plot(recall, precision, lw=2, label=f"AP = {average_precision:.4f}")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title(title)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.legend(loc="lower left")
  plt.grid(alpha=0.2)
  plt.show()

  return average_precision


def compare_average_precision(
  models: Mapping[str, Any],
  X: np.ndarray,
  y_true: np.ndarray,
  title: str = "Precision-Recall Curves and Average Precision",
) -> Dict[str, float]:
  """Plot PR curves for multiple fitted models and compare their AP values."""
  if len(models) == 0:
    raise ValueError("models must contain at least one fitted model.")

  ap_scores: Dict[str, float] = {}
  plt.figure(figsize=(14, 6))
  ax_curve = plt.subplot(1, 2, 1)
  ax_bar = plt.subplot(1, 2, 2)

  for name, model in models.items():
    y_target, y_score, is_binary = _precision_recall_inputs(model, X, y_true)
    if is_binary:
      precision, recall, _ = precision_recall_curve(y_target, y_score)
      ap = float(average_precision_score(y_target, y_score))
    else:
      precision, recall, _ = precision_recall_curve(y_target.ravel(), y_score.ravel())
      ap = float(average_precision_score(y_target, y_score, average="micro"))

    ap_scores[name] = ap
    ax_curve.plot(recall, precision, lw=2, label=f"{name} (AP = {ap:.4f})")

  ax_curve.set_xlabel("Recall")
  ax_curve.set_ylabel("Precision")
  ax_curve.set_title("Precision-Recall Curves")
  ax_curve.set_xlim([0.0, 1.0])
  ax_curve.set_ylim([0.0, 1.05])
  ax_curve.grid(alpha=0.2)
  ax_curve.legend(loc="lower left")

  names = list(ap_scores.keys())
  values = [ap_scores[name] for name in names]
  bars = ax_bar.bar(names, values, color=plt.cm.tab10(np.linspace(0, 1, len(names))))
  ax_bar.set_ylim([0.0, 1.05])
  ax_bar.set_ylabel("Average Precision")
  ax_bar.set_title("Average Precision by Model")
  ax_bar.grid(axis="y", alpha=0.2)
  ax_bar.tick_params(axis="x", rotation=30)

  for bar, value in zip(bars, values):
    ax_bar.text(
      bar.get_x() + bar.get_width() / 2,
      value + 0.02,
      f"{value:.4f}",
      ha="center",
      va="bottom",
      fontsize=9,
    )

  plt.suptitle(title)
  plt.tight_layout()
  plt.show()

  return ap_scores


# McNemar's test for paired correctness
def mcnemar_paired(
  y_true: np.ndarray,
  y_pred_a: np.ndarray,
  y_pred_b: np.ndarray,
) -> Dict[str, Any]:
  """McNemar on paired correctness (any number of classes)."""
  correct_a = np.asarray(y_pred_a) == np.asarray(y_true)
  correct_b = np.asarray(y_pred_b) == np.asarray(y_true)
  n01 = int(np.sum(~correct_a & correct_b))
  n10 = int(np.sum(correct_a & ~correct_b))
  n_disc = n01 + n10
  if n_disc == 0:
    return {
      "n01": n01,
      "n10": n10,
      "statistic": np.nan,
      "pvalue": 1.0,
      "method": "exact",
    }
  if n_disc < 25:
    p = float(binomtest(min(n01, n10), n_disc, 0.5, alternative="two-sided").pvalue,)
    return {
      "n01": n01, 
      "n10": n10, 
      "statistic": np.nan, 
      "pvalue": p, 
      "method": "exact",
    }
  stat = (abs(n01 - n10) - 1) ** 2 / n_disc
  p = float(1 - chi2.cdf(stat, df=1))
  return {
    "n01": n01,
    "n10": n10,
    "statistic": stat,
    "pvalue": p,
    "method": "chi2_cc",
  }


# Reliability diagram and ECE for multiclass classification
def reliability_diagram_multiclass(
  y_true: np.ndarray, 
  probs: np.ndarray, 
  class_labels: np.ndarray, 
  n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
  y_true = np.asarray(y_true)
  probs = np.asarray(probs, dtype=float)
  class_labels = np.asarray(class_labels)

  pred_idx = np.argmax(probs, axis=1)
  confidences = probs[np.arange(probs.shape[0]), pred_idx]
  predictions = class_labels[pred_idx]
  correctness = (predictions == y_true).astype(float)

  bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
  bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
  counts = np.zeros(n_bins, dtype=int)
  conf_mean = np.full(n_bins, np.nan, dtype=float)
  acc_mean = np.full(n_bins, np.nan, dtype=float)
  ece = 0.0

  bin_idx = np.digitize(confidences, bin_edges[1:-1], right=False)
  for idx in range(n_bins):
    mask = bin_idx == idx
    counts[idx] = int(np.sum(mask))
    if counts[idx] == 0:
      continue
    conf_mean[idx] = float(np.mean(confidences[mask]))
    acc_mean[idx] = float(np.mean(correctness[mask]))
    ece += (counts[idx] / y_true.shape[0]) * abs(acc_mean[idx] - conf_mean[idx])

  return bin_centers, counts, conf_mean, acc_mean, float(ece)

