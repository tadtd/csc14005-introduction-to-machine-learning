from .base import Regression
from .linear_reg import LinearRegression
from .ridge_reg import RidgeRegression
from .lasso_reg import LassoRegression
from .elastic_net import ElasticNetRegression
from .perceptron_reg import PerceptronRegression
from .kernel_ridge import KernelRidgeRegression
from .robust_reg import RobustRegression
from .gaussian_process_reg import GaussianProcessRegression
from .bayesian_reg import BayesianRegression
from .wls_reg import FeasibleWeightedLeastSquaresRegression, WeightedLeastSquaresRegression
from .feature_map_reg import FeatureMapRegressor

__all__ = [
    "Regression",
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
    "PerceptronRegression",
    "KernelRidgeRegression",
    "RobustRegression",
    "GaussianProcessRegression",
    "BayesianRegression",
    "WeightedLeastSquaresRegression",
    "FeasibleWeightedLeastSquaresRegression",
    "FeatureMapRegressor",
]
