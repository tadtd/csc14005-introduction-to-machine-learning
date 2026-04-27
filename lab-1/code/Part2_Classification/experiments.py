import numpy as np
import pandas as pd
from itertools import combinations
from models import LogisticRegression as LR

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


def fisher_ratio_ranking(
  X_train_df: pd.DataFrame,
  X_train_scaled: np.ndarray,
  y_train: np.ndarray,
) -> pd.DataFrame:
  """Return Fisher ratio ranking for each feature in a multiclass setting."""
  feature_names = list(X_train_df.columns)
  classes, counts = np.unique(y_train, return_counts=True)
  overall_mean = X_train_scaled.mean(axis=0)
  fisher_scores = []

  for j in range(X_train_scaled.shape[1]):
    between_class = 0.0
    within_class = 0.0

    for cls, n_c in zip(classes, counts):
      x_c = X_train_scaled[y_train == cls, j]
      mu_c = x_c.mean()
      var_c = x_c.var(ddof=1)

      between_class += n_c * (mu_c - overall_mean[j]) ** 2
      within_class += n_c * var_c

    fisher_scores.append(between_class / (within_class + 1e-12))

  fisher_df = pd.DataFrame(
    {
      'feature': feature_names,
      'fisher_ratio': fisher_scores,
    }
  )
  return fisher_df.sort_values('fisher_ratio', ascending=False).reset_index(drop=True)
