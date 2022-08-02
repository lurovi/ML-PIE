import numpy as np
import math
from pymoo.core.problem import ElementwiseProblem

from gp.tree.PrimitiveTree import PrimitiveTree


class SampleProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0)
        self.target_function = lambda x: math.sin(x) + math.sin(x ** 2) + math.sin(x) * math.cos(x) + np.random.normal(
            scale=0.1, size=1)[0]
        self.interval_lower_bound = -3
        self.interval_upper_bound = 3
        self.step = 0.1

    def _evaluate(self, x, out, *args, **kwargs):
        total_error = 0
        total_evaluations = 0
        tree = x[0]
        for i in np.arange(self.interval_lower_bound, self.interval_upper_bound, self.step):
            target = self.target_function(i)
            value = tree.compile([i])
            square_error = (value - target) ** 2
            total_error = total_error + square_error
            total_evaluations = total_evaluations + 1
        mse = total_error / total_evaluations
        tree_size = tree.number_of_nodes()
        out["F"] = np.array([mse, tree_size], dtype=float)
