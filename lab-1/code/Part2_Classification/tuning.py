import numpy as np
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from models import Classification

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
