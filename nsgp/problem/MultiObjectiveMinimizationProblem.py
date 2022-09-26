import threading
from typing import List
from copy import deepcopy
from pymoo.core.problem import Problem
import numpy as np

from nsgp.evaluation.TreeEvaluator import TreeEvaluator


class MultiObjectiveMinimizationProblem(Problem):
    def __init__(self, evaluators: List[TreeEvaluator], mutex: threading.Lock = None):
        super().__init__(n_var=1, n_obj=len(evaluators), n_ieq_constr=0, n_eq_constr=0)
        self.__evaluators = deepcopy(evaluators)
        self.__number_of_evaluators = len(self.__evaluators)
        self.__mutex = mutex

    def _evaluate(self, x, out, *args, **kwargs):
        if self.__mutex is not None:
            with self.__mutex:
                self._eval(x, out)
        else:
            self._eval(x, out)

    def _eval(self, x, out, *args, **kwargs):
        out["F"] = np.empty((len(x), self.__number_of_evaluators), dtype=np.float32)
        for i in range(len(x)):
            tree = x[i, 0]
            for j in range(self.__number_of_evaluators):
                out["F"][i, j] = self.__evaluators[j].evaluate(tree)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle mutex
        del state["_" + self.__class__.__name__ + "__mutex"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__mutex = None
