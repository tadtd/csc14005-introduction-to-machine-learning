from models import Classification
from typing import Mapping, Any
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def plot_learning_curve(
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

def plot_loss_curve(
  loss_history: list[float] | np.ndarray,
  title: str = "Training Loss Curve",
  xlabel: str = "Epoch / Iteration",
  ylabel: str = "Cross-Entropy Loss",
  log_scale: bool = False,
) -> None:
  """Plot the training loss curve over the number of epochs or iterations.

  Parameters
  ----------
  loss_history
    A list or 1D array of loss values recorded during training.
  title
    The title of the plot.
  xlabel
    Label for the x-axis.
  ylabel
    Label for the y-axis.
  log_scale
    Whether to plot the y-axis in logarithmic scale.
  """
  losses = np.asarray(loss_history)
  if losses.ndim != 1:
    raise ValueError("loss_history must be a 1D array or list.")
    
  epochs = np.arange(1, len(losses) + 1)
  
  plt.figure(figsize=(8, 6))
  plt.plot(epochs, losses, lw=2, color="blue", label="Training Loss")
  
  if log_scale:
    plt.yscale("log")
    
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(alpha=0.3)
  plt.legend(loc="upper right")
  plt.tight_layout()
  plt.show()


def compare_loss_curves(
  models_losses: Mapping[str, Any],
  title: str = "Training Loss Curves Comparison",
  xlabel: str = "Epoch / Iteration",
  ylabel: str = "Cross-Entropy Loss",
  log_scale: bool = False,
) -> None:
  """Plot and compare multiple loss curves.
  
  Parameters
  ----------
  models_losses
    A dictionary mapping model names to their corresponding loss histories.
  title, xlabel, ylabel, log_scale
    Plot customization parameters.
  """
  if not models_losses:
    raise ValueError("models_losses must contain at least one entry.")
    
  plt.figure(figsize=(10, 6))
  
  for name, loss_history in models_losses.items():
    losses = np.asarray(loss_history)
    epochs = np.arange(1, len(losses) + 1)
    plt.plot(epochs, losses, lw=2, label=name)
    
  if log_scale:
    plt.yscale("log")
    
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.show()


def choose_lambda_cv(
  model: Classification,
  X: np.ndarray,
  y: np.ndarray,
  lambdas: list[float] | np.ndarray,
  param_name: str = "lam",
  k: int = 5,
  scoring: str = "accuracy",
  average: str = "weighted",
  random_state: int = 42,
  fit_kwargs: dict | None = None,
) -> float:
  """
  Choose the best regularization parameter using stratified k-fold CV.
  
  Parameters
  ----------
  model : Classification
    The unfitted model to evaluate.
  X, y : np.ndarray
    The input features and target labels.
  lambdas : list[float] | np.ndarray
    The array of candidate lambda (regularization) values to test.
  param_name : str
    The name of the attribute in the model that represents the regularization.
    Defaults to 'lam' (used in KernelLogisticRegression). For LogisticRegression,
    you might pass 'l2_penalty' or 'l1_penalty'.
  k : int
    The number of folds for StratifiedKFold (default 5).
  scoring : str
    The metric to track (e.g., 'accuracy', 'f1-score').
  average : str
    The averaging method for the metrics (e.g., 'weighted', 'macro').
  fit_kwargs: dict
    Any extra arguments passed to fold_model.fit(X, y).
    For LogisticRegression, pass `fit_kwargs={'solver': 'l2'}`.
    
  Returns
  -------
  float
    The lambda value that achieved the highest cross-validation score.
  """
  X = np.asarray(X)
  y = np.asarray(y)
  fit_kwargs = fit_kwargs or {}
  
  splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
  best_score = -np.inf
  best_lam = None
  
  print(f"Starting Stratified CV (k={k}) for '{param_name}' over {len(lambdas)} parameters...")
  
  for lam_val in lambdas:
    fold_scores = []
    
    for train_idx, val_idx in splitter.split(X, y):
      fold_model = deepcopy(model)
      # Dynamically assign the specified attribute for regularization
      setattr(fold_model, param_name, lam_val)
      
      # Fit model with any necessary extra parameters (like solvers)
      fold_model.fit(X[train_idx], y[train_idx], **fit_kwargs)
      
      y_pred = fold_model.predict(X[val_idx])
      metrics = fold_model.evaluate(y_pred, y[val_idx], average=average)
      fold_scores.append(metrics[scoring])
      
    mean_score = float(np.mean(fold_scores))
    print(f"  {param_name}={lam_val:<10.6g} \t Mean {scoring}: {mean_score:.4f}")
    
    if mean_score > best_score:
      best_score = mean_score
      best_lam = lam_val
      
  print(f"Best {param_name}: {best_lam} (Score: {best_score:.4f})")
  return best_lam

def plot_2d_decision_boundary(models_dict, X, y, title_prefix=""):
    # Determine grid boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    n_models = len(models_dict)
    cols = min(2, n_models)
    rows = (n_models + 1) // 2 if n_models > 2 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        if not hasattr(model, "predict"):
            continue
        try:
            # Predict over the grid
            Z = model.predict(grid)
            
            # Map predictions to integer indices if they are original class labels
            unique_classes = model.classes_
            Z_idx = np.zeros_like(Z, dtype=int)
            for idx, cls in enumerate(unique_classes):
                Z_idx[Z == cls] = idx
            
            Z_idx = Z_idx.reshape(xx.shape)
            
            # Convert y to numerical indices as well for scatter colors
            y_idx = np.zeros_like(y, dtype=int)
            for idx, cls in enumerate(unique_classes):
                y_idx[y == cls] = idx
                
            ax.contourf(xx, yy, Z_idx, alpha=0.4, cmap=cmap_light)
            ax.scatter(X[:, 0], X[:, 1], c=y_idx, cmap=cmap_bold, edgecolor='k', s=20)
            ax.set_title(f"{title_prefix} {name}")
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        except Exception as e:
            ax.set_title(f"{name} (Failed to plot)")
            print(f"Failed for {name}: {e}")
            
    # Remove unused axes
    for i in range(len(models_dict), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()
from models import LogisticRegression as LR
from itertools import combinations

def ovr_predict_with_logistic(X_train, y_train, X_test, learning_rate, eps, max_iter):
  classes = np.unique(y_train)
  models_ovr = {}
  class_scores = []

  for cls in classes:
    y_binary = (y_train == cls).astype(int)
    model = LR(learning_rate=learning_rate, eps=eps, max_iter=max_iter)
    model.fit(X_train, y_binary)
    models_ovr[cls] = model
    class_scores.append(model.predict_proba(X_test)[:, 1])

  class_scores = np.column_stack(class_scores)
  y_pred = classes[np.argmax(class_scores, axis=1)]
  return y_pred, models_ovr

def ovo_predict_with_logistic(X_train, y_train, X_test, learning_rate, eps, max_iter):
  classes = np.unique(y_train)
  class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

  votes = np.zeros((X_test.shape[0], len(classes)), dtype=int)
  confidences = np.zeros((X_test.shape[0], len(classes)), dtype=float)
  models_ovo = {}

  for cls_i, cls_j in combinations(classes, 2):
    pair_mask = (y_train == cls_i) | (y_train == cls_j)
    X_pair = X_train[pair_mask]
    y_pair = y_train[pair_mask]

    y_pair_binary = (y_pair == cls_j).astype(int)
    model = LR(learning_rate=learning_rate, eps=eps, max_iter=max_iter)
    model.fit(X_pair, y_pair_binary)
    models_ovo[(cls_i, cls_j)] = model

    pair_prob_j = model.predict_proba(X_test)[:, 1]
    pair_pred = np.where(pair_prob_j >= 0.5, cls_j, cls_i)

    idx_i, idx_j = class_to_idx[cls_i], class_to_idx[cls_j]
    votes[np.arange(X_test.shape[0]), [class_to_idx[p] for p in pair_pred]] += 1
    confidences[:, idx_i] += (1.0 - pair_prob_j)
    confidences[:, idx_j] += pair_prob_j

  max_votes = votes.max(axis=1, keepdims=True)
  is_tie = (votes == max_votes).sum(axis=1) > 1
  pred_idx = np.argmax(votes, axis=1)

  for i in np.where(is_tie)[0]:
    tie_mask = votes[i] == votes[i].max()
    tie_candidates = np.where(tie_mask)[0]
    best_tie_idx = tie_candidates[np.argmax(confidences[i, tie_candidates])]
    pred_idx[i] = best_tie_idx

  y_pred = classes[pred_idx]
  return y_pred, models_ovo

def plot_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
    plt.title(title)
    plt.show()
