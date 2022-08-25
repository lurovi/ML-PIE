import inspect
import math
from random import random

from genepro import node_impl
from genepro.node import Node
from genepro.node_impl import Constant, IfThenElse, Feature
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from nsgp.genepro_integration import *


class SimpleFunctionProblem(ElementwiseProblem):

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
        for v in np.arange(self.interval_lower_bound, self.interval_upper_bound, self.step):
            target = self.target_function(v)
            array = np.array([[v]])
            value = tree.get_output(array)
            square_error = (value - target) ** 2
            total_error = total_error + square_error
            total_evaluations = total_evaluations + 1
        mse = total_error / total_evaluations
        tree_size = tree.get_n_nodes()

        out["F"] = np.array([mse, tree_size], dtype=float)


node_classes = [c[1] for c in inspect.getmembers(node_impl, inspect.isclass)]
internal_nodes = list()
for node_cls in node_classes:
    # handle Features and Constants separately (also, avoid base class Node and IfThenElse)
    if node_cls == Node or node_cls == Feature or node_cls == Constant or node_cls == IfThenElse:
        continue
    node_obj = node_cls()
    internal_nodes.append(node_obj)

# 10 random constants and 1 feature
leaf_nodes = list()
leaf_nodes.append(Feature(0))
for i in range(10):
    leaf_nodes.append(Constant(random()))

tree_sampling = TreeSampling(internal_nodes, leaf_nodes, 4)
tree_crossover = TreeCrossover()
tree_mutation = TreeMutation(internal_nodes, leaf_nodes)

algorithm = NSGA2(pop_size=11,
                  sampling=tree_sampling,
                  crossover=tree_crossover,
                  mutation=tree_mutation,
                  eliminate_duplicates=DuplicateTreeElimination()
                  )

res = minimize(SimpleFunctionProblem(),
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=False)

Scatter().add(res.F).show()
