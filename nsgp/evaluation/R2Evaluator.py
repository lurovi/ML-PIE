import numpy as np
from sklearn.metrics import r2_score

from genepro.util import compute_linear_scaling

from genepro.node import Node
from nsgp.evaluation.TreeEvaluator import TreeEvaluator


class R2Evaluator(TreeEvaluator):
    def __init__(self, X: np.ndarray, y: np.ndarray = None, linear_scaling: bool = True, negate: bool = False):
        super().__init__()
        if y is None:
            raise AttributeError("Labels must be set.")
        if y.shape[0] != X.shape[0]:
            raise AttributeError(
                f"The number of observations in X is {X.shape[0]} and it is different from the number of observations in y, i.e., {y.shape[0]}.")
        if len(y.shape) != 1:
            raise AttributeError(
                f"y must be one-dimensional. The number of dimensions that have been detected in y are, on the contrary, {len(y.shape)}.")
        self.__X = X
        self.__y = y
        self.__linear_scaling = linear_scaling
        self.__sign = -1.0 if negate else 1.0

    def evaluate(self, tree: Node, **kwargs) -> float:
        res: np.ndarray = np.core.umath.clip(tree(self.__X), -1e+10, 1e+10)
        slope, intercept = 1.0, 0.0
        if self.__linear_scaling:
            slope, intercept = compute_linear_scaling(self.__y, res)
            slope = np.core.umath.clip(slope, -1e+10, 1e+10)
            intercept = np.core.umath.clip(intercept, -1e+10, 1e+10)
            res = intercept + np.core.umath.clip(slope * res, -1e+10, 1e+10)
            res = np.core.umath.clip(res, -1e+10, 1e+10)
        r2: float = self.__sign * r2_score(res, self.__y)
        return r2
