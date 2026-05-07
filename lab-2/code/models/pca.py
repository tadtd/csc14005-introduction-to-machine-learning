from __future__ import annotations
import numpy as np
from typing import Any, Optional, Dict
from .base import BaseDR

class PCA(BaseDR):
  """Principal Component Analysis."""

  def __init__(
    self,
    n_components: int = 2,
    **kwargs: Any
  ) -> None:
    super().__init__(n_components, **kwargs)
    self.components_: Optional[np.ndarray] = None
    self.explained_variance_: Optional[np.ndarray] = None
  
  def _fit(self, X: np.ndarray) -> None:
    raise NotImplementedError("Subclasses must implement _fit")
  
  def _transform(self, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Subclasses must implement _transform")
  
  def _inverse_transform(self, Y: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Subclasses must implement _inverse_transform")
  
  def get_params(self) -> Dict[str, Any]:
    return {**super().get_params()}