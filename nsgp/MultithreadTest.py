import threading

from functools import partial

from torch import nn

from deeplearn.model.MLPNet import MLPNet
from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer
from genepro import node_impl
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.optimize import minimize
import time
import numpy as np
import random
import torch

from nsgp.encoder.CountsEncoder import CountsEncoder
from nsgp.callback.PopulationAccumulator import PopulationAccumulator
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.RegressionProblemWithNeuralEstimate import RegressionProblemWithNeuralEstimate
from nsgp.structure.TreeStructure import TreeStructure
from util.PicklePersist import PicklePersist


def example_of_difficult_target_for_regression(x):
    return ((x[0] ** 3 + x[1] ** 2) ** 2) * np.log(abs(x[2] * np.e) + 1e-9) - x[8] * np.pi - x[4] * 5 + np.sqrt(
        abs(x[7]) * np.e) + np.cos(x[5] * 2) + np.sin(x[6] * 7) - np.exp(x[9]) / 1000.0 + (np.pi * x[0]) ** 3


def example_of_difficult_target_for_binary_classification(x):
    if x[9] ** 2 + np.cos(x[0] * np.pi + np.sin(x[1] * np.e)) + max(
            np.log(abs(x[2] * x[8] * np.pi) + 1e-9) + np.sqrt(abs(x[3] * np.e) + 1e-9), (x[4] * x[5]) / 2.0) + np.exp(
        x[7]) + 2 * np.exp(x[2]) > np.cos(x[3]) - np.sin(x[6]) + np.exp(x[5]) / (abs(x[7] * x[9]) + 1e-9) - (
            np.pi * x[0]) ** 2:
        return 1
    else:
        return 0


def run_optimization(local_mutex, local_tree_encoder, local_trainer, local_population_storage):
    algorithm = NSGA2(pop_size=20,
                      sampling=tree_sampling,
                      crossover=tree_crossover,
                      mutation=tree_mutation,
                      eliminate_duplicates=duplicates_elimination
                      )
    wind_speed = PicklePersist.decompress_pickle("../exps/windspeed/wind_dataset_split.pbz2")
    start = time.time()
    res = minimize(
        RegressionProblemWithNeuralEstimate(
            wind_speed["training"][0], wind_speed["training"][1], mutex=local_mutex, tree_encoder=local_tree_encoder,
            interpretability_estimator=local_trainer
        ),
        algorithm,
        ('n_gen', 10),
        seed=10,
        verbose=False,
        callback=PopulationAccumulator(population_storage=local_population_storage),
        save_history=True)
    end = time.time()
    print((end - start))
    pops = res.algorithm.callback.population_storage

    print(len(pops))


def run_feedback(local_mutex, local_tree_encoder, local_trainer, local_population_storage):
    updater = InterpretabilityEstimateUpdater(individuals=local_population_storage, mutex=local_mutex,
                                              interpretability_estimator=local_trainer, encoder=local_tree_encoder)
    time.sleep(1)
    while True:
        updater.collect_feedback()


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    duplicates_elimination_little_data = np.random.uniform(0.0, 1.0, size=(10, 7))

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

    structure = TreeStructure(internal_nodes, 7, 5, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0),
                              normal_distribution_parameters=normal_distribution_parameters)
    setting = TreeSetting(structure, duplicates_elimination_little_data)
    tree_sampling = setting.get_sampling()
    tree_crossover = setting.get_crossover()
    tree_mutation = setting.get_mutation()
    duplicates_elimination = setting.get_duplicates_elimination()
    tree_encoder = CountsEncoder(structure)

    mlp_net = MLPNet(nn.ReLU(), nn.Identity(), tree_encoder.size(), 1, [220, 110, 25])
    trainer = OnlineTwoPointsCompareTrainer(mlp_net, device)

    mutex = threading.Lock()
    population_storage = set()
    estimator = []

    # Setting random seed to allow scientific reproducibility
    t1 = threading.Thread(
        target=partial(run_optimization, local_mutex=mutex, local_tree_encoder=tree_encoder, local_trainer=trainer,
                       local_population_storage=population_storage)).start()
    t2 = threading.Thread(
        target=partial(run_feedback, local_mutex=mutex, local_tree_encoder=tree_encoder, local_trainer=trainer,
                       local_population_storage=population_storage)).start()
