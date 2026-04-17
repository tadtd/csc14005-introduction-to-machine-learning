from models import Classification
# from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot as plt

def learning_curve(
  model: Classification,
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
  scoring: str = "accuracy",
  ) -> None:
  """
  Return training and validation accuracies for each training set size in ``train_sizes``.
  """
  # cam on anh, anh do mixi
  train_errors, val_errors = [], []
  subset_sizes = (train_sizes * len(X_train)).astype(int)
  for size in subset_sizes:
    X_subset, y_subset = X_train[:size], y_train[:size]
    model.fit(X=X_subset, y=y_subset)
    y_train_pred = model.predict(X=X_subset)
    y_val_pred = model.predict(X=X_val)
    train_acc = model.evaluate(y_train_pred, y_subset, average="weighted")["accuracy"]
    val_acc = model.evaluate(y_val_pred, y_val, average="weighted")["accuracy"]
    train_errors.append(train_acc)
    val_errors.append(val_acc)
  
  plt.figure(figsize=(10, 6))
  plt.plot(subset_sizes, train_errors, "o-", label="Training")
  plt.plot(subset_sizes, val_errors, "o-", label="Validation")
  plt.xlabel("Training Set Size")
  plt.ylabel("Accuracy")
  plt.title("Learning Curve")
  plt.legend()
  plt.show()

def loss_curve(
  model: Classification,
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  max_iter: int = 1000,
) -> None:
  """
  Return training and validation losses for each iteration of gradient descent.
  Only works for models that implement ``self.loss_history`` during fitting.
  """
  raise NotImplementedError("loss_curve is not implemented yet.")