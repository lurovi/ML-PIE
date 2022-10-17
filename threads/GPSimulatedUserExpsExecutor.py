import random
import threading
from copy import deepcopy
import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.operators.selection.tournament import TournamentSelection
from torch import nn
from deeplearn.model.MLPNet import MLPNet
from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer
from deeplearn.trainer.TwoPointsCompareTrainerFactory import TwoPointsCompareTrainerFactory
from exps.DatasetGenerator import DatasetGenerator
from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from nsgp.callback.PopulationAccumulator import PopulationAccumulator

from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.RegressionProblemWithNeuralEstimate import RegressionProblemWithNeuralEstimate
from nsgp.sampling.GroundTruthCollector import GroundTruthCollector
from nsgp.sampling.PairChooserFactory import PairChooserFactory

from nsgp.structure.TreeStructure import TreeStructure
from threads.MlPieAutomaticRun import MlPieAutomaticRun
from threads.OptimizationThread import OptimizationThread

from typing import List, Dict


class GPSimulatedUserExpsExecutor:
    def __init__(self, folder_name: str,
                 data_name: str,
                 structure: TreeStructure,
                 ground_truths: List[GroundTruthComputer],
                 dataset: Dict,
                 duplicates_elimination_little_data: np.ndarray,
                 device: torch.device,
                 data_generator: DatasetGenerator,
                 verbose: bool = False):
        self.__verbose = verbose
        self.__folder_name = folder_name
        self.__data_name = data_name
        self.__data_generator = deepcopy(data_generator)
        self.__device = device
        self.__structure = deepcopy(structure)
        self.__ground_truths = deepcopy(ground_truths)
        self.__dataset = deepcopy(dataset)
        self.__duplicates_elimination_little_data = deepcopy(duplicates_elimination_little_data)
        self.__ground_truths_names = {k.get_name(): k for k in self.__ground_truths}

    def execute_gp_run(self, optimization_seed: int, pop_size: int, num_gen: int, encoding_type: str, ground_truth_type: str, sampler_factory: PairChooserFactory, warmup: str = None) -> bool:
        num_offsprings = pop_size
        termination = ("n_gen", num_gen)
        tournament_selection = TournamentSelection(func_comp=binary_tournament, pressure=2)
        setting = TreeSetting(self.__structure, self.__duplicates_elimination_little_data, crossover_prob=0.9,
                              mutation_prob=0.6)
        tree_sampling = setting.get_sampling()
        tree_crossover = setting.get_crossover()
        tree_mutation = setting.get_mutation()
        duplicates_elimination = setting.get_duplicates_elimination()
        algorithm = NSGA2(pop_size=pop_size,
                          n_offsprings=num_offsprings,
                          selection=tournament_selection,
                          sampling=tree_sampling,
                          crossover=tree_crossover,
                          mutation=tree_mutation,
                          eliminate_duplicates=duplicates_elimination
                          )

        if warmup is not None:
            warmup_plot = warmup[0].upper() + warmup[1:]
            warmup_plot = warmup_plot.replace("_", " ")
            warmup_data = self.__data_generator.get_warm_up_data(encoding_type, warmup)
            pretrainer_factory = TwoPointsCompareTrainerFactory(False, 1)
        else:
            warmup_plot = "No Warm-up"
            warmup_data = None
            pretrainer_factory = None

        # shared parameters
        tree_encoder = self.__structure.get_encoder(encoding_type)
        random.seed(optimization_seed)
        np.random.seed(optimization_seed)
        torch.manual_seed(optimization_seed)
        mlp_net = MLPNet(nn.ReLU(), nn.Identity(), tree_encoder.size(), 1, [220, 110, 25], dropout_prob=0.25)
        interpretability_estimator = OnlineTwoPointsCompareTrainer(mlp_net, self.__device,
                                                                   warmup_trainer_factory=pretrainer_factory,
                                                                   warmup_dataset=warmup_data)
        mutex = threading.Lock()
        population_storage = set()

        # optimization thread creation
        problem = RegressionProblemWithNeuralEstimate(self.__dataset["training"][0],
                                                      self.__dataset["training"][1],
                                                      mutex=mutex,
                                                      tree_encoder=tree_encoder,
                                                      interpretability_estimator=interpretability_estimator
                                                      )

        # feedback thread creation
        feedback_collector = GroundTruthCollector(self.__ground_truths_names[ground_truth_type])
        interpretability_estimate_updater = InterpretabilityEstimateUpdater(individuals=population_storage,
                                                                            mutex=mutex,
                                                                            interpretability_estimator=interpretability_estimator,
                                                                            encoder=tree_encoder,
                                                                            pair_chooser=sampler_factory)

        callback = PopulationAccumulator(population_storage=population_storage)
        optimization_thread = OptimizationThread(
            optimization_algorithm=algorithm,
            problem=problem,
            termination=termination,
            seed=optimization_seed,
            callback=callback,
            verbose=self.__verbose
        )
        parameters = {"seed": optimization_seed,
                      "pop_size": pop_size, "num_gen": num_gen, "num_offsprings": num_offsprings,
                      "encoder_type": encoding_type, "ground_truth_type": ground_truth_type,
                      "sampling": sampler_factory.create(1).get_string_repr(), "warm-up": warmup_plot,
                      "data": self.__data_name}
        run_id = parameters["data"]+"-"+parameters["encoder_type"]+"-"+parameters["ground_truth_type"]+"-"+parameters["sampling"]+"-"+parameters["warm-up"]+"-"+"GPSU"+"_"+str(optimization_seed)
        # thread execution
        automatic_run = MlPieAutomaticRun(
            run_id=run_id, path=self.__folder_name+"/",
            optimization_thread=optimization_thread,
            interpretability_estimate_updater=interpretability_estimate_updater,
            feedback_collector=feedback_collector,
            parameters=parameters
        )
        random.seed(optimization_seed)
        np.random.seed(optimization_seed)
        torch.manual_seed(optimization_seed)
        automatic_run.run_automatically(delay=5)
        print("\n"+run_id+"\n")
        return True
