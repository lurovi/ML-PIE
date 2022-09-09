import numpy as np
from pymoo.core.problem import ElementwiseProblem


class RegressionProblem(ElementwiseProblem):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, n_eq_constr=0)
        if y.shape[0] != X.shape[0]:
            raise AttributeError(f"The number of observations in X is {X.shape[0]} and it is different from the number of observations in y, i.e., {y.shape[0]}.")
        if len(y.shape) != 1:
            raise AttributeError(f"y must be one-dimensional. The number of dimensions that have been detected in y are, on the contrary, {len(y.shape)}.")
        self.__n_records: int = X.shape[0]
        self.__X: np.ndarray = X
        self.__y: np.ndarray = y

    def _evaluate(self, x, out, *args, **kwargs):
        tree = x[0]
        res: np.ndarray = tree(self.__X)
        mse: float = np.square(np.clip(res - self.__y, -1e+20, 1e+20)).sum()/float(self.__n_records)
        if mse > 1e+20:
            mse = 1e+20
        out["F"] = np.array([mse, tree.get_n_nodes()], dtype=np.float32)
