import collections
import inspect
from functools import partial

from exps.groundtruth.NumNodesNegComputer import NumNodesNegComputer
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

from nsgp.evaluation.GroundTruthEvaluator import GroundTruthEvaluator
from nsgp.evaluation.MSEEvaluator import MSEEvaluator
from nsgp.evolution.GPWithNSGA2 import GPWithNSGA2
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.BinaryClassificationProblem import BinaryClassificationProblem
from nsgp.problem.MultiObjectiveMinimizationElementWiseProblem import MultiObjectiveMinimizationElementWiseProblem
from nsgp.problem.RegressionProblem import RegressionProblem
from nsgp.problem.SimpleFunctionProblem import SimpleFunctionProblem
from nsgp.structure.TreeStructure import TreeStructure
from util.PicklePersist import PicklePersist


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
    duplicates_elimination_little_data = np.random.uniform(-1.0, 1.0, size=(10, 7))

    internal_nodes = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                 node_impl.UnaryMinus(), node_impl.Power(), node_impl.Square(), node_impl.Cube(),
                 node_impl.Sqrt(), node_impl.Exp(),
                 node_impl.Log(), node_impl.Sin(),
                 node_impl.Cos()]

    normal_distribution_parameters = [(0, 1), (0, 1), (0, 3), (0, 8), (0, 0.5),
                                      (0, 15), (0, 5), (0, 8), (0, 20), (0, 30),
                                      (0, 30), (0, 23), (0, 23),
                                      (0, 0.8), (0, 0.8), (0, 0.8), (0, 0.8), (0, 0.8),
                                      (0, 0.8), (0, 0.8),
                                      (0, 0.5)]
    wind_speed = PicklePersist.decompress_pickle("D:/shared_folder/python_projects/ML-PIE/exps/windspeed/wind_dataset_split.pbz2")

    structure = TreeStructure(internal_nodes, 7, 5, ephemeral_func=partial(np.random.uniform, low=-5.0, high=5.0), normal_distribution_parameters=normal_distribution_parameters)
    evaluators = [MSEEvaluator(wind_speed["training"][0], wind_speed["training"][1]), GroundTruthEvaluator(NumNodesNegComputer(), True)]
    gp = GPWithNSGA2(structure, evaluators, pop_size=20, num_gen=50, duplicates_elimination_data=duplicates_elimination_little_data)
    res = gp.run_minimization(seed=1)
    print(res["executionTimeInHours"])
    Scatter().add(res["result"].F).show()
