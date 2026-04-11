import numpy as np
from scipy.stats import chi2, binomtest

from typing import Dict, Any, Tuple


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
