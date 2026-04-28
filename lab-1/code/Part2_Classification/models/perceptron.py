import numpy as np
from itertools import count

from .base import Classification

class Perceptron(Classification):
  """
  Perceptron binary classification algorithm fit with the standard perceptron learning rule.
  """

  def __init__(
    self,
    learning_rate: float = 1.0,
    max_iter: int | None = 1000,
    min_iter: int = 1,
    early_stopping_patience: int = 100,
    early_stopping_tol: float = 1e-12,
  ):
    super().__init__()
    self.learning_rate = learning_rate
    self.max_iter = max_iter
    self.min_iter = min_iter
    self.early_stopping_patience = early_stopping_patience
    self.early_stopping_tol = early_stopping_tol

  def fit(self, X: np.ndarray, y: np.ndarray) -> None:
    self.classes_ = np.unique(y)
    if len(self.classes_) != 2:
      raise ValueError(f"Perceptron is meant for binary classification; got {len(self.classes_)} classes.")
      
    n_samples, n_features = X.shape
    
    # Map labels to {-1, 1}
    y_binary = np.where(y == self.classes_[1], 1, -1)
    
    self.theta = np.zeros(n_features + 1)
    X_aug = self._augment(X)
    
    self.loss_history_ = []

    prev_loss = np.inf
    stagnation = 0
    
    iterations = range(self.max_iter) if self.max_iter is not None else count()
    for epoch in iterations:
      misclassified = 0
      loss = 0.0
      
      for i in range(n_samples):
        fx = np.dot(X_aug[i], self.theta)
        
        if y_binary[i] * fx <= 0:
          self.theta += self.learning_rate * y_binary[i] * X_aug[i]
          misclassified += 1
          loss += -y_binary[i] * fx
          
      self.loss_history_.append(loss)
      loss_change = abs(prev_loss - loss)
      if epoch + 1 >= self.min_iter and loss_change < self.early_stopping_tol:
        stagnation += 1
      else:
        stagnation = 0
          
      if misclassified == 0 and epoch + 1 >= self.min_iter:
        break
      if (
        self.early_stopping_patience > 0
        and epoch + 1 >= self.min_iter
        and stagnation >= self.early_stopping_patience
      ):
        break

      prev_loss = loss
    else:
      pass

  def predict(self, X: np.ndarray) -> np.ndarray:
    if getattr(self, "theta", None) is None:
      raise RuntimeError("Call fit before predict.")
      
    X_aug = self._augment(X)
    scores = X_aug @ self.theta
    
    # If >= 0 -> class 1, else class 0
    return np.where(scores >= 0, self.classes_[1], self.classes_[0])
                  
  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    if getattr(self, "theta", None) is None:
      raise RuntimeError("Call fit before predict_proba.")
      
    X_aug = self._augment(X)
    scores = X_aug @ self.theta
    
    # Since standard Perceptron does not output probabilities, we can just pseudo-probability
    # based on confidence or strictly 0/1. 
    # For ROC AUC, we can output the distance to the hyperplane mapped to [0, 1] using sigmoid.
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        
    probs = sigmoid(scores)
    return np.column_stack([1 - probs, probs])
