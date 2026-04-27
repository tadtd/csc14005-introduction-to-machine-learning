from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ClassificationNotebookConfig:
    seed: int = 42
    alpha: float = 0.05
    data_path: str = '../../data/raw/dry-bean-dataset/Dry_Bean_Dataset.xlsx'
    target_col: str = 'Class'
    test_size: float = 0.2
    val_size: float = 0.1
    binary_classes: tuple[int, int] = (0, 2)
    binary_feature_indices: tuple[int, int] = (0, 1)
    kfold_splits: int = 5
    reliability_bins: int = 10

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
    max_iter: Optional[int] = 5000
    class_weight: Optional[Union[str, dict]] = None

@dataclass
class LDAConfig:
    reg: float = 1e-4

@dataclass
class LogisticRegressionConfig:
    learning_rate: float = 1e-4
    eps: float = 1e-4
    max_iter: Optional[int] = 5000
    prior_precision: float = 1.0
    penalize_bias: bool = False
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    class_weight: Optional[Union[str, dict]] = None

@dataclass
class PerceptronConfig:
    learning_rate: float = 1.0
    max_iter: Optional[int] = 5000

@dataclass
class ProbitRegressionConfig:
    learning_rate: float = 1e-3
    eps: float = 1e-4
    max_iter: Optional[int] = 5000
    class_weight: Optional[Union[str, dict]] = None

@dataclass
class QDAConfig:
    reg: float = 1e-4

@dataclass
class SoftmaxRegressionConfig:
    learning_rate: float = 1e-4
    eps: float = 1e-4
    max_iter: Optional[int] = 5000
    prior_precision: float = 1.0
    penalize_bias: bool = False
    class_weight: Optional[Union[str, dict]] = None
