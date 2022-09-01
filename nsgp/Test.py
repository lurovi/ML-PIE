import inspect
import math
from random import random

import numpy as np

from genepro import node_impl
from genepro.node import Node
from genepro.node_impl import Constant, IfThenElse, Feature
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from nsgp.genepro_integration import *
from nsgp.operator.TreeSetting import TreeSetting
from util.TreeGrammarStructure import TreeGrammarStructure


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
            square_error = np.clip(value - target, -1.340780792993396e+150, 1.340780792993396e+150) ** 2
            if square_error > 1.3407e+150:
                total_error = 1.3407e+150
                total_evaluations = 1.0
                break
            total_error = total_error + square_error
            total_evaluations = total_evaluations + 1
        mse = total_error / total_evaluations
        tree_size = tree.get_n_nodes()

        out["F"] = np.array([mse, tree_size], dtype=float)


if __name__ == "__main__":
    node_classes = [c[1] for c in inspect.getmembers(node_impl, inspect.isclass)]
    internal_nodes = list()
    for node_cls in node_classes:
        # handle Features and Constants separately (also, avoid base class Node and IfThenElse)
        if node_cls == Node or node_cls == Feature or node_cls == Constant or node_cls == IfThenElse:
            continue
        node_obj = node_cls()
        internal_nodes.append(node_obj)

    structure = TreeGrammarStructure(internal_nodes, 1, 7, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0))
    setting = TreeSetting(structure, np.array([[0.23],[12],[0.45],[0.45],[1.23],[2.4],[1.8],[0.90]]))
    tree_sampling = setting.get_sampling()
    tree_crossover = setting.get_crossover()
    tree_mutation = setting.get_mutation()
    duplicates_elimination = setting.get_duplicates_elimination()

    algorithm = NSGA2(pop_size=20,
                      sampling=tree_sampling,
                      crossover=tree_crossover,
                      mutation=tree_mutation,
                      eliminate_duplicates=duplicates_elimination
                      )

    res = minimize(SimpleFunctionProblem(),
                   algorithm,
                   ('n_gen', 30),
                   seed=5,
                   verbose=True)

    Scatter().add(res.F).show()
