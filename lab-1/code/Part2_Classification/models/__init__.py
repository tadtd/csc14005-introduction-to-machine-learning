from .base import Classification
from .logistic_reg import LogisticRegression
from .softmax_reg import SoftmaxRegression
from .lda import LDA
from .qda import QDA
from .gnb import GaussianNaiveBayes


__all__ = [
  "Classification",
  "LogisticRegression",
  "SoftmaxRegression",
  "LDA",
  "QDA",
  "GaussianNaiveBayes",
]