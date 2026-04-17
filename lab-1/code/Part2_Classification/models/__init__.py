from .base import Classification
from .kernel_log import KernelLogisticRegression
from .logistic_reg import LogisticRegression
from .probit import ProbitRegression
from .softmax_reg import SoftmaxRegression
from .lda import LDA
from .qda import QDA
from .gnb import GaussianNaiveBayes


__all__ = [
  "Classification",
  "KernelLogisticRegression",
  "LogisticRegression",
  "ProbitRegression",
  "SoftmaxRegression",
  "LDA",
  "QDA",
  "GaussianNaiveBayes",
]