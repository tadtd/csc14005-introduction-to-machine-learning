from .base import Regression
import numpy as np

class LinearRegression(Regression):
  def __init__(
    self,
    learning_rate: float = 1e-3,
    eps: float = 1e-6,
    max_iter: int | None = None,
  ) -> None:
    super().__init__()
    self.learning_rate = learning_rate
    self.eps = eps
    self.max_iter = max_iter
  
  def fit(self, X: np.ndarray, y: np.ndarray) -> None:
    raise NotImplementedError("fit is not implemented yet.")
  