import random
import threading
import time

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual
from pymoo.visualization.scatter import Scatter
from torch import nn

from deeplearn.model.MLPNet import MLPNet
from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer
from exps.groundtruth.InterpretabilityShapeComputer import InterpretabilityShapeComputer
from genepro import node_impl
from genepro.node import Node
from nsgp.callback.PopulationAccumulator import PopulationAccumulator
from nsgp.encoder.CountsEncoder import CountsEncoder
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.RegressionProblemWithNeuralEstimate import RegressionProblemWithNeuralEstimate
from nsgp.sampling.GroundTruthCollector import GroundTruthCollector
from nsgp.sampling.RandomChooserOnline import RandomChooserOnline
from nsgp.sampling.StringFromTerminalCollector import StringFromTerminalCollector
from nsgp.structure.TreeStructure import TreeStructure
from threads.FeedbackThread import FeedbackThread
from threads.OptimizationThread import OptimizationThread
from util.PicklePersist import PicklePersist

if __name__ == '__main__':
    print("Main thread starting")

    # settings
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tree parameters
    duplicates_elimination_little_data = np.random.uniform(0.0, 1.0, size=(10, 7))
    internal_nodes = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                      node_impl.UnaryMinus(), node_impl.Power(), node_impl.Square(), node_impl.Cube(),
                      node_impl.Sqrt(), node_impl.Exp(), node_impl.Log(), node_impl.Sin(), node_impl.Cos()]
    normal_distribution_parameters = [(0, 1), (0, 1), (0, 3), (0, 8), (0, 0.5), (0, 15), (0, 5), (0, 8), (0, 20),
                                      (0, 30), (0, 30), (0, 23), (0, 23), (0, 0.8), (0, 0.8), (0, 0.8), (0, 0.8),
                                      (0, 0.8), (0, 0.8), (0, 0.8), (0, 0.5)]
    structure = TreeStructure(internal_nodes, 7, 5, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0),
                              normal_distribution_parameters=normal_distribution_parameters)
    setting = TreeSetting(structure, duplicates_elimination_little_data)
    tree_sampling = setting.get_sampling()
    tree_crossover = setting.get_crossover()
    tree_mutation = setting.get_mutation()
    duplicates_elimination = setting.get_duplicates_elimination()

    # shared parameters
    tree_encoder = CountsEncoder(structure)
    mlp_net = MLPNet(nn.ReLU(), nn.Identity(), tree_encoder.size(), 1, [220, 110, 25])
    interpretability_estimator = OnlineTwoPointsCompareTrainer(mlp_net, device)
    mutex = threading.Lock()
    population_storage = set()

    # optimization thread creation
    algorithm = NSGA2(pop_size=20,
                      sampling=tree_sampling,
                      crossover=tree_crossover,
                      mutation=tree_mutation,
                      eliminate_duplicates=duplicates_elimination
                      )
    dataset = PicklePersist.decompress_pickle("../exps/windspeed/wind_dataset_split.pbz2")
    problem = RegressionProblemWithNeuralEstimate(dataset["training"][0], dataset["training"][1], mutex=mutex,
                                                  tree_encoder=tree_encoder,
                                                  interpretability_estimator=interpretability_estimator
                                                  )
    termination = ('n_gen', 5)
    optimization_seed = seed
    callback = PopulationAccumulator(population_storage=population_storage)
    optimization_thread = OptimizationThread(
        optimization_algorithm=algorithm,
        problem=problem,
        termination=termination,
        seed=optimization_seed,
        callback=callback
    )

    # feedback thread creation
    pair_chooser = RandomChooserOnline()
    feedback_collector = GroundTruthCollector(InterpretabilityShapeComputer())
    interpretability_estimate_updater = InterpretabilityEstimateUpdater(individuals=population_storage, mutex=mutex,
                                                                        interpretability_estimator=interpretability_estimator,
                                                                        encoder=tree_encoder, pair_chooser=pair_chooser,
                                                                        feedback_collector=feedback_collector)
    feedback_thread = FeedbackThread(interpretability_estimate_updater=interpretability_estimate_updater, delay=0.3)

    # thread execution
    optimization_thread.start()
    callback.population_non_empty.wait()
    feedback_thread.start()

    # thread termination
    optimization_thread.join()
    print("Optimization is over")
    feedback_thread.stop()
    feedback_thread.join()
    print("Feedback collection is stopped")

    history = optimization_thread.result.history
    for generation in history:
        for individual in generation.opt:
            individual: Individual = individual
            tree: Node = individual.X[0]
            print(
                f"Accuracy: {individual.F[0]}\t Interpretability: {individual.F[1]} \t Tree: {tree.get_readable_repr()}")
        print()
