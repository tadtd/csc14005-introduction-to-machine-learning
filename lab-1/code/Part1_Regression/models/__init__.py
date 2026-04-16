from .base import Regression
from .linear_reg import LinearRegression
from .ridge_reg import RidgeRegression
from .lasso_reg import LassoRegression
from .elastic_net import ElasticNetRegression
from .perceptron_reg import PerceptronRegression
from .kernel_ridge import KernelRidgeRegression

__all__ = [
    "Regression",
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
    "PerceptronRegression",
    "KernelRidgeRegression",
]
