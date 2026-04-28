import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from models import Classification
from models import LogisticRegression

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
    
    if mean_score > best_score:
      best_score = mean_score
      best_lam = lam_val
      
  return best_lam


def tune_logistic_regularization(
  X: np.ndarray,
  y: np.ndarray,
  *,
  base_kwargs: dict,
  lambda_grid: list[float] | np.ndarray,
  solvers: tuple[str, ...] = ("l1", "l2"),
  k: int = 5,
  random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
  """Tune L1/L2 logistic penalties with stratified k-fold CV."""
  X = np.asarray(X)
  y = np.asarray(y)
  splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

  rows: list[dict] = []
  best: dict[str, dict | None] = {solver: None for solver in solvers}

  for solver in solvers:
    for lam in lambda_grid:
      fold_scores = []
      for tr_idx, va_idx in splitter.split(X, y):
        kwargs = dict(base_kwargs)
        kwargs["l1_penalty"] = float(lam) if solver in ("l1", "elastic_net") else 0.0
        kwargs["l2_penalty"] = float(lam) if solver in ("l2", "elastic_net") else 0.0
        model = LogisticRegression(**kwargs)
        model.fit(X[tr_idx], y[tr_idx], solver=solver)
        y_pred = model.predict(X[va_idx])
        metrics = model.evaluate(y_pred, y[va_idx], average="weighted")
        fold_scores.append(float(metrics["f1-score"]))

      score_mean = float(np.mean(fold_scores))
      score_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
      row = {
        "penalty": solver.upper(),
        "lambda": float(lam),
        "cv_f1_mean": score_mean,
        "cv_f1_std": score_std,
      }
      rows.append(row)
      if best[solver] is None or score_mean > best[solver]["cv_f1_mean"]:
        best[solver] = row

  tuning_df = pd.DataFrame(rows)
  best_lambdas = {solver: float(best[solver]["lambda"]) for solver in solvers if best[solver] is not None}
  return tuning_df, best_lambdas
