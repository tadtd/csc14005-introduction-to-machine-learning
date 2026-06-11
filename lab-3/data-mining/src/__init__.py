"""
Data Mining: Tensor Decomposition Module
=========================================

Part III of "Foundations of Tensor Computations for AI (NeurIPS 2025)"

This package provides implementations of:
- CP-ALS (Canonical Polyadic - Alternating Least Squares) from scratch
- CP-OPT (Gradient-based CP via L-BFGS-B)
- PARAFAC2 (for irregular slices)
- EEG data loading and tensor construction
- Evaluation metrics and visualization

All CP-ALS core operations (unfold, khatri-rao, MTTKRP) are implemented
from scratch using NumPy only.
"""

from . import cp_als
from . import cp_opt
from . import parafac2
from . import data
from . import evaluate
from . import visualize

__version__ = "0.1.0"
__all__ = ["cp_als", "cp_opt", "parafac2", "data", "evaluate", "visualize"]
