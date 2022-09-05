import inspect

from genepro import node_impl
from genepro.node import Node
from genepro.node_impl import Constant, IfThenElse, Feature
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import numpy as np
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.SimpleFunctionProblem import SimpleFunctionProblem
from nsgp.util.TreeGrammarStructure import TreeGrammarStructure


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
                   seed=1,
                   verbose=True)

    Scatter().add(res.F).show()
