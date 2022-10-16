import math

import numpy as np
from pymoo.core.problem import ElementwiseProblem


class SimpleFunctionProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0)
        self.__target_function = lambda x: math.sin(x) + math.sin(x ** 2) + math.sin(x) * math.cos(x) + np.random.normal(
            scale=0.1, size=1)[0]
        self.__interval_lower_bound = -3
        self.__interval_upper_bound = 3
        self.__step = 0.1

    def _evaluate(self, x, out, *args, **kwargs):
        total_error = 0
        total_evaluations = 0
        tree = x[0]
        for v in np.arange(self.__interval_lower_bound, self.__interval_upper_bound, self.__step):
            target = self.__target_function(v)
            array = np.array([[v]])
            value = tree.get_output(array)[0]
            square_error = np.core.umath.clip(value - target, -1.340780792993396e+150, 1.340780792993396e+150) ** 2
            if square_error > 1.3407e+150:
                total_error = 1.3407e+150
                total_evaluations = 1.0
                break
            total_error = total_error + square_error
            total_evaluations = total_evaluations + 1
        mse = total_error / total_evaluations
        tree_size = tree.get_n_nodes()

        out["F"] = np.array([mse, tree_size], dtype=float)
