import numpy as np

from genepro.node import Node
from nsgp.evaluation.TreeEvaluator import TreeEvaluator


class MSEEvaluator(TreeEvaluator):
    def __init__(self, X: np.ndarray, y: np.ndarray = None, linear_scaling: bool = True):
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

    def evaluate(self, tree: Node) -> float:
        res: np.ndarray = tree(self.__X)
        if self.linear_scaling:
            slope, intercept = compute_linear_scaling(self.__y, res)
            res = intercept + slope * res
        mse: float = np.square(np.clip(res - self.__y, -1e+20, 1e+20)).sum() / float(len(self.__y))
        if mse > 1e+20:
            mse = 1e+20
        return mse
