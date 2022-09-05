import numpy as np
from pymoo.core.problem import ElementwiseProblem


class RegressionProblem(ElementwiseProblem):
    def __init__(self, n_var: int, n_obj: int, n_ieq_constr: int, X: np.ndarray, y: np.ndarray):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr)
        self.__n_records = X.shape[0]
        self.__X = X
        self.__y = y

    def _evaluate(self, x, out, *args, **kwargs):
        res = x(self.__X)
        mse = np.square(np.clip(res - self.__y, -1e+100, 1e+100)).sum()/float(self.__n_records)
        return np.array([mse, x.get_n_nodes()], dtype=np.float32)

