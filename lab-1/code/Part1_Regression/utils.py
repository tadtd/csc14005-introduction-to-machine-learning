from models import Regression
import numpy as np

def learning_curve(
  model: Regression,
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  train_sizes: np.linspace = np.linspace(0.1, 1.0, 10),
) -> None:
  """
  Return training and validation losses for each training set size in ``train_sizes``.
  """
  raise NotImplementedError("learning_curve is not implemented yet.")

def loss_curve(
  model: Regression,
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  # max_iter: int = 1000,
) -> None:
  """
  Return training and validation losses for each iteration of gradient descent.
  Only works for models that implement ``self.loss_history`` during fitting.
  """
  raise NotImplementedError("loss_curve is not implemented yet.")