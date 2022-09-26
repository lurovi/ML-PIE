from nsgp.evaluation.TreeEvaluator import TreeEvaluator
from genepro.node import Node
import numpy as np


class BinaryMSEEvaluator(TreeEvaluator):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        super().__init__()
        if y is None:
            raise AttributeError("Labels must be set.")
        if y.shape[0] != X.shape[0]:
            raise AttributeError(
                f"The number of observations in X is {X.shape[0]} and it is different from the number of observations in y, i.e., {y.shape[0]}.")
        if len(y.shape) != 1:
            raise AttributeError(
                f"y must be one-dimensional. The number of dimensions that have been detected in y are, on the contrary, {len(y.shape)}.")
        if sorted(np.unique(y).tolist()) != [0, 1]:
            raise AttributeError(
                f"y must contains only categorical labels, and since this is a binary classification problem, there must be 2 classes: 0 and 1, so y must contains only 0 and 1 (at least one observation for each class).")
        self.__X = X
        self.__y = y

    def evaluate(self, tree: Node) -> float:
        res: np.ndarray = np.where(tree(self.__X) > 0, 1, 0)
        mse: float = np.square(res - self.__y).sum() / float(len(self.__y))
        return mse
