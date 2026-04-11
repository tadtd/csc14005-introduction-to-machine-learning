import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

from matplotlib import pyplot as plt


class Regression(ABC):
  def __init__(self) -> None:
    self.theta: np.ndarray | None = None

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

  def evaluate(
    self,
    y_pred: np.ndarray,
    y_true: np.ndarray,
  ) -> Dict[str, Any]:
    raise NotImplementedError("evaluate is not implemented yet.")