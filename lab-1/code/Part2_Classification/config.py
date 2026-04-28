from dataclasses import dataclass
from typing import Optional, Union

SEED = 42
ALPHA = 0.05
DATA_PATH = '../../data/raw/dry-bean-dataset/Dry_Bean_Dataset.xlsx'
TARGET_COL = 'Class'
TEST_SIZE = 0.2
VAL_SIZE = 0.1
BINARY_CLASSES = (0, 2)
BINARY_FEATURE_INDICES = (0, 1)
KFOLD_SPLITS = 5
N_BINS = 10

@dataclass
class GaussianNaiveBayesConfig:
    reg: float = 1e-4

@dataclass
class KernelLogisticRegressionConfig:
    kernel: str = 'rbf'
    gamma: float = 0.5
    lam: float = 1e-4
    learning_rate: float = 1e-2
    eps: float = 1e-4
    max_iter: Optional[int] = 20_000
    min_iter: int = 100
    early_stopping_patience: int = 100
    early_stopping_tol: float = 1e-6
    class_weight: Optional[Union[str, dict]] = None

@dataclass
class LDAConfig:
    reg: float = 1e-4

@dataclass
class LogisticRegressionConfig:
    learning_rate: float = 1e-2
    eps: float = 1e-4
    max_iter: Optional[int] = 20_000
    min_iter: int = 100
    early_stopping_patience: int = 100
    early_stopping_tol: float = 1e-6
    prior_precision: float = 1.0
    penalize_bias: bool = False
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    class_weight: Optional[Union[str, dict]] = None

@dataclass
class PerceptronConfig:
    learning_rate: float = 1.0
    max_iter: Optional[int] = 20_000
    min_iter: int = 1
    early_stopping_patience: int = 100
    early_stopping_tol: float = 1e-6

@dataclass
class ProbitRegressionConfig:
    learning_rate: float = 1e-2
    eps: float = 1e-4
    max_iter: Optional[int] = 20_000
    min_iter: int = 100
    early_stopping_patience: int = 100
    early_stopping_tol: float = 1e-6
    class_weight: Optional[Union[str, dict]] = None

@dataclass
class QDAConfig:
    reg: float = 1e-4

@dataclass
class SoftmaxRegressionConfig:
    learning_rate: float = 1e-2
    eps: float = 1e-4
    max_iter: Optional[int] = 20_000
    min_iter: int = 100
    early_stopping_patience: int = 100
    early_stopping_tol: float = 1e-6
    prior_precision: float = 1.0
    penalize_bias: bool = False
    class_weight: Optional[Union[str, dict]] = None
