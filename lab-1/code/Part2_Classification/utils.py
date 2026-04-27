from models import Classification
from typing import Mapping, Any
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

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



def plot_2d_decision_boundary(models_dict, X, y, title_prefix="", class_name_map=None):
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
            if len(unique_classes) == 2:
                label_0 = unique_classes[0]
                label_1 = unique_classes[1]
                if class_name_map is not None:
                    label_0 = f"{unique_classes[0]} ({class_name_map.get(unique_classes[0], unique_classes[0])})"
                    label_1 = f"{unique_classes[1]} ({class_name_map.get(unique_classes[1], unique_classes[1])})"
                handles = [
                    Line2D([0], [0], marker='o', color='w', label=f'class {label_0}',
                           markerfacecolor='#FF0000', markeredgecolor='k', markersize=7),
                    Line2D([0], [0], marker='o', color='w', label=f'class {label_1}',
                           markerfacecolor='#0000FF', markeredgecolor='k', markersize=7),
                ]
                ax.legend(handles=handles, loc='best')
        except Exception as e:
            ax.set_title(f"{name} (Failed to plot)")
            print(f"Failed for {name}: {e}")
            
    # Remove unused axes
    for i in range(len(models_dict), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()

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
