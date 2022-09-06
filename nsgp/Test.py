import collections
import inspect

from genepro import node_impl
from genepro.node import Node
from genepro.node_impl import Constant, IfThenElse, Feature
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import time
import numpy as np
import random
import torch
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.BinaryClassificationProblem import BinaryClassificationProblem
from nsgp.problem.RegressionProblem import RegressionProblem
from nsgp.problem.SimpleFunctionProblem import SimpleFunctionProblem
from nsgp.util.TreeGrammarStructure import TreeGrammarStructure


def example_of_difficult_target_for_regression(x):
    return ((x[0]**3 + x[1]**2)**2)*np.log(abs(x[2]*np.e) + 1e-9) - x[8]*np.pi - x[4]*5 + np.sqrt(abs(x[7])*np.e) + np.cos(x[5]*2) + np.sin(x[6]*7) - np.exp(x[9])/1000.0 + (np.pi*x[0])**3


def example_of_difficult_target_for_binary_classification(x):
    if x[9]**2 + np.cos(x[0]*np.pi + np.sin(x[1]*np.e)) + max(np.log(abs(x[2]*x[8]*np.pi) + 1e-9) + np.sqrt(abs(x[3]*np.e) + 1e-9), (x[4]*x[5])/2.0) + np.exp(x[7]) + 2*np.exp(x[2]) > np.cos(x[3]) - np.sin(x[6]) + np.exp(x[5])/(abs(x[7]*x[9]) + 1e-9) - (np.pi*x[0])**2:
        return 1
    else:
        return 0


if __name__ == "__main__":
    # Setting random seed to allow scientific reproducibility
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_data_X = np.random.uniform(0.0, 1.0, size=(5000, 10))
    regression_y = np.array([example_of_difficult_target_for_regression(random_data_X[i]) for i in range(random_data_X.shape[0])])
    binary_classification_y = np.array([example_of_difficult_target_for_binary_classification(random_data_X[i]) for i in range(random_data_X.shape[0])])
    print(collections.Counter(binary_classification_y))
    duplicates_elimination_little_data = np.random.uniform(0.0, 1.0, size=(10, 10))
    duplicates_elimination_little_data_0 = np.array([[0.23], [12], [0.45], [0.45], [1.23], [2.4], [1.8], [0.90]])

    node_classes = [c[1] for c in inspect.getmembers(node_impl, inspect.isclass)]
    internal_nodes = list()
    for node_cls in node_classes:
        # handle Features and Constants separately (also, avoid base class Node and IfThenElse)
        if node_cls == Node or node_cls == Feature or node_cls == Constant or node_cls == IfThenElse:
            continue
        node_obj = node_cls()
        internal_nodes.append(node_obj)

    structure = TreeGrammarStructure(internal_nodes, 10, 7, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0))
    setting = TreeSetting(structure, duplicates_elimination_little_data)
    tree_sampling = setting.get_sampling()
    tree_crossover = setting.get_crossover()
    tree_mutation = setting.get_mutation()
    duplicates_elimination = setting.get_duplicates_elimination()

    algorithm = NSGA2(pop_size=50,
                      sampling=tree_sampling,
                      crossover=tree_crossover,
                      mutation=tree_mutation,
                      eliminate_duplicates=duplicates_elimination
                      )
    start = time.time()
    res = minimize(BinaryClassificationProblem(random_data_X, binary_classification_y),
                   algorithm,
                   ('n_gen', 30),
                   seed=10,
                   verbose=True,
                   save_history=True)
    end = time.time()
    print((end - start))
    Scatter().add(res.F).show()
