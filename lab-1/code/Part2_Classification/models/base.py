import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

from matplotlib import pyplot as plt
from sklearn.metrics import (
  auc,
  classification_report,
  confusion_matrix,
  precision_recall_fscore_support,
  roc_curve,
)
from sklearn.preprocessing import label_binarize


class Classification(ABC):
  def __init__(self) -> None:
    self.theta: np.ndarray | None = None
    self.classes_: np.ndarray | None = None

  # adding a bias term is common to all models
  @staticmethod
  def _augment(X: np.ndarray) -> np.ndarray:
    """Append a ones column to X: (n, d) -> (n, d+1)."""
    return np.c_[X, np.ones(X.shape[0])]

  @abstractmethod
  def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
    **kwargs: Any,
  ) -> None:
    """Fit the model to the training data."""

  @abstractmethod
  def predict(
    self,
    X: np.ndarray,
  ) -> np.ndarray:
    """Predict class labels for the given input data."""

  @abstractmethod
  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities (or scores) for the given input data."""

  def evaluate(
    self,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    average: Literal["micro", "macro", "weighted", "binary"] = "weighted",
    sample_weight: Optional[np.ndarray] = None,
  ) -> Dict[str, Any]:
    """
    Return accuracy, aggregated precision-recall-F1 (by ``average``), and the full report.
    Argument order matches common notebook usage: predictions first, then ground truth.

    Parameters
    ----------
    average
      Same idea as ``sklearn.metrics`` *average*: ``'macro'``, ``'weighted'`` (default),
      ``'micro'``, or ``'binary'`` (binary problems only). Sets top-level ``precision``,
      ``recall``, and ``f1-score``.
    sample_weight
      Optional per-sample weights passed to the underlying sklearn metrics.
    """
    report = classification_report(
      y_true=y_true,
      y_pred=y_pred,
      output_dict=True,
      zero_division=0,
      sample_weight=sample_weight,
    )
    if average in ("macro", "weighted"):
      primary = report[f"{average} avg"]
    elif average in ("micro", "binary"):
      p, r, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
        sample_weight=sample_weight,
      )
      primary = {
        "precision": float(p),
        "recall": float(r),
        "f1-score": float(f),
      }
    else:
      raise ValueError(
        "average must be 'macro', 'weighted', 'micro', or 'binary', "
        f"got {average!r}"
      )
    return {
      "accuracy": report["accuracy"],
      "precision": primary["precision"],
      "recall": primary["recall"],
      "f1-score": primary["f1-score"],
      "macro_avg": {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1-score": report["macro avg"]["f1-score"],
      },
      "weighted_avg": {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"],
      },
      "classification_report": report,
    }

  def plot_confusion_matrix(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False,
    title: str = "Confusion Matrix",
  ) -> None:
    """Plot the confusion matrix (optionally row-normalized)."""
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    if normalize:
      row_sums = cm.sum(axis=1, keepdims=True)
      cm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
      )

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, origin="upper")
    plt.title(title)
    plt.colorbar()

    labels = np.unique(np.concatenate((y_true, y_pred)))
    tick_marks = np.arange(len(labels))

    plt.xticks(ticks=tick_marks, labels=labels)
    plt.yticks(ticks=tick_marks, labels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
        val = cm[i, j]
        text = f"{val:{fmt}}" if normalize else f"{int(round(val))}"
        plt.text(
          j,
          i,
          text,
          ha="center",
          va="center",
          color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.show()

  def plot_roc_and_auc_curve(
    self,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC and AUC Curves",
  ) -> None:
    """
    Plot ROC curve and report AUC.
    For binary problems, pass shape (n,) positive-class scores or (n, 2) probabilities.
    For multiclass, pass shape (n, n_classes) probabilities (micro-averaged ROC).
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if y_pred_proba.ndim == 1:
      y_score = y_pred_proba
    elif y_pred_proba.shape[1] == 2:
      y_score = y_pred_proba[:, 1]
    else:
      y_score = y_pred_proba

    classes = np.unique(y_true)
    n_classes = len(classes)

    if n_classes < 2:
      raise ValueError("ROC requires at least two distinct classes in y_true.")

    if n_classes == 2:
      fpr, tpr, _ = roc_curve(y_true, y_score)
    else:
      y_bin = label_binarize(y_true, classes=classes)
      if y_score.ndim != 2 or y_score.shape[1] != n_classes:
        raise ValueError(
          "Multiclass ROC expects y_pred_proba of shape (n_samples, n_classes)."
        )
      fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
      fpr,
      tpr,
      color="darkorange",
      lw=2,
      label=f"ROC curve (AUC = {roc_auc:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
