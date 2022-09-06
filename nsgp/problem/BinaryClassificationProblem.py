import numpy as np
from pymoo.core.problem import ElementwiseProblem


class BinaryClassificationProblem(ElementwiseProblem):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, n_eq_constr=0)
        if y.shape[0] != X.shape[0]:
            raise AttributeError(f"The number of observations in X is {X.shape[0]} and it is different from the number of observations in y, i.e., {y.shape[0]}.")
        if len(y.shape) != 1:
            raise AttributeError(f"y must be one-dimensional. The number of dimensions that have been detected in y are, on the contrary, {len(y.shape)}.")
        if sorted(np.unique(y).tolist()) != [0, 1]:
            raise AttributeError(f"y must contains only categorical labels, and since this is a binary classification problem, there must be 2 classes: 0 and 1, so y must contains only 0 and 1 (at least one observation for each class).")
        self.__n_records: int = X.shape[0]
        self.__X: np.ndarray = X
        self.__y: np.ndarray = y

    def _evaluate(self, x, out, *args, **kwargs):
        tree = x[0]
        res: np.ndarray = np.where(tree(self.__X) > 0, 1, 0)
        mse: float = np.square(res - self.__y).sum()/float(self.__n_records)
        out["F"] = np.array([mse, tree.get_n_nodes()], dtype=np.float32)
