import random
from functools import partial
from typing import List

import pandas as pd
import torch
import numpy as np
import torch.multiprocessing as mp
from pymoo.core.result import Result

from exps.DatasetGenerator import DatasetGenerator
from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from exps.groundtruth.LevelWiseWeightsSumNegComputer import LevelWiseWeightsSumNegComputer
from exps.groundtruth.MathElasticModelComputer import MathElasticModelComputer
from exps.groundtruth.NodeWiseWeightsSumNegComputer import NodeWiseWeightsSumNegComputer
from exps.groundtruth.NumNodesNegComputer import NumNodesNegComputer
from exps.groundtruth.WeightsSumNegComputer import WeightsSumNegComputer
from genepro import node_impl
from nsgp.encoder.CountsEncoder import CountsEncoder
from nsgp.encoder.LevelWiseCountsEncoder import LevelWiseCountsEncoder
from nsgp.encoder.OneHotEncoder import OneHotEncoder
from nsgp.structure.TreeStructure import TreeStructure
from threads.MlPieRun import MlPieRun
from util.PicklePersist import PicklePersist


class ExpsUtil:

    @staticmethod
    def set_random_seed(seed: int = None) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def create_structure(dataset_path: str):
        # Setting torch to use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True)
        # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        ExpsUtil.set_random_seed(100)

        dataset = PicklePersist.decompress_pickle(dataset_path)
        n_features: int = dataset["training"][0].shape[1]
        duplicates_elimination_little_data_num_points: int = 5
        duplicates_elimination_little_data = np.random.uniform(-5.0, 5.0, size=(duplicates_elimination_little_data_num_points, n_features))
        internal_nodes = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                          node_impl.UnaryMinus(), node_impl.Power(), node_impl.Square(), node_impl.Cube(),
                          node_impl.Sqrt(), node_impl.Exp(), node_impl.Log(), node_impl.Sin(), node_impl.Cos()]
        normal_distribution_parameters = [(0, 1), (0, 1), (0, 3), (0, 8), (0, 0.5), (0, 15), (0, 5), (0, 8), (0, 20),
                                          (0, 30), (0, 30), (0, 23), (0, 23)] + [(0, 0.8)] * n_features + [(0, 0.5)]
        structure = TreeStructure(internal_nodes, n_features, 5, ephemeral_func=partial(np.random.uniform, -5.0, 5.0),
                                  normal_distribution_parameters=normal_distribution_parameters)
        structure.register_encoders([CountsEncoder(structure, True, 100), LevelWiseCountsEncoder(structure, True, 100),
                                     OneHotEncoder(structure)])
        ground_truths = [NumNodesNegComputer(), MathElasticModelComputer(),
                         WeightsSumNegComputer(structure, 100),
                         LevelWiseWeightsSumNegComputer(structure, 100), NodeWiseWeightsSumNegComputer(structure, 100)]

        return structure, ground_truths, dataset, duplicates_elimination_little_data

    @staticmethod
    def create_dataset_generator_with_warmup(folder_name: str, data_name: str, structure: TreeStructure,
                                             ground_truths: List[GroundTruthComputer]):
        train_size = 1250
        validation_size = 370
        test_size = 250
        data_generator = DatasetGenerator(folder_name, structure, train_size, validation_size, test_size, 101)

        data_generator.generate_tree_encodings(True)

        data_generator.generate_ground_truth(ground_truths)

        for e in structure.get_encoding_type_strings():
            for i in range(2):
                data_generator.create_dataset_warm_up_from_encoding_ground_truth(20, e, ground_truths[i], 102)

        #data_generator.create_dataset_warm_up_from_csv("tree_data_1" + "/FeynmanEquationsWarmUp.csv", "feynman", 20)

        data_generator.persist(data_name + "_datasets_generator")
        return data_generator

    @staticmethod
    def save_pareto_fronts_from_result_to_csv(folder_name: str,
                                              result: Result, seed: int,
                                              pop_size: int, num_gen: int, num_offsprings: int,
                                              dataset: str, groundtruth: str,
                                              encoder_type: str = "not_spcified",
                                              sampling: str = "not_specified",
                                              warmup: str = "not_specified"
                                              ) -> None:
        parameters = {"seed": seed,
                      "pop_size": pop_size, "num_gen": num_gen, "num_offsprings": num_offsprings,
                      "encoder_type": encoder_type, "ground_truth_type": groundtruth,
                      "sampling": sampling, "warm-up": warmup,
                      "data": dataset}
        run_id = parameters["data"] + "-" + parameters["ground_truth_type"] + "-" + "GPT" + "_" + str(seed)
        # write optimization file
        generations, parsable_trees, latex_trees, accuracies, interpretabilities, uncertainties = MlPieRun.parse_optimization_history(
            result.history, [[1.0]*pop_size]*num_gen)
        best_data = pd.DataFrame(
            list(zip(generations, parsable_trees, latex_trees, accuracies, interpretabilities, uncertainties)),
            columns=['generation', 'parsable_tree', 'latex_tree', 'accuracy', 'interpretability', 'uncertainties'])

        # update dataframes
        for k in parameters.keys():
            best_data[k] = parameters[k]

        # save files
        best_data.to_csv(path_or_buf=folder_name + "/" + "best-" + run_id + ".csv")
