import random

from functools import partial

from deeplearn.dataset.RandomSamplerOnline import RandomSamplerOnline
from deeplearn.dataset.RandomSamplerOnlineFactory import RandomSamplerOnlineFactory
from deeplearn.dataset.UncertaintySamplerOnline import UncertaintySamplerOnline
from deeplearn.dataset.UncertaintySamplerOnlineDistanceEmbeddings import UncertaintySamplerOnlineDistanceEmbeddings
from deeplearn.dataset.UncertaintySamplerOnlineDistanceEmbeddingsFactory import \
    UncertaintySamplerOnlineDistanceEmbeddingsFactory
from deeplearn.dataset.UncertaintySamplerOnlineFactory import UncertaintySamplerOnlineFactory
from exps.DatasetGenerator import DatasetGenerator
from exps.ExpsExecutor import ExpsExecutor
from exps.groundtruth.InterpretabilityShapeComputer import InterpretabilityShapeComputer
from exps.groundtruth.LevelWiseWeightsSumNegComputer import LevelWiseWeightsSumNegComputer
from exps.groundtruth.LispExprHashComputer import LispExprHashComputer
from exps.groundtruth.MathElasticModelComputer import MathElasticModelComputer
from exps.groundtruth.NodeWiseWeightsSumNegComputer import NodeWiseWeightsSumNegComputer
from exps.groundtruth.NumNodesNegComputer import NumNodesNegComputer
from exps.groundtruth.WeightsSumNegComputer import WeightsSumNegComputer
from genepro import node_impl as node_impl

from deeplearn.model.MLPNet import *

from nsgp.encoder.CountsEncoder import CountsEncoder
from nsgp.encoder.LevelWiseCountsEncoder import LevelWiseCountsEncoder
from nsgp.encoder.OneHotEncoder import OneHotEncoder
from nsgp.structure.TreeStructure import TreeStructure

import torch.multiprocessing as mp
import pandas as pd
import numpy as np

from util.PicklePersist import PicklePersist
from util.PlotGenerator import PlotGenerator

pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def set_random_seed(seed: int = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    # Setting torch to use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    set_random_seed(100)
    n_features = 6
    max_depth = 4
    operators = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                          node_impl.Cube(),
                          node_impl.Log(), node_impl.Max()]

    normal_distribution_parameters = [(0, 1), (0, 1), (0, 3), (0, 8),
                                      (0, 8),
                                      (0, 30), (0, 15)] + [(0, 0.8)] * n_features + [(0, 0.5)]

    structure = TreeStructure(operators, n_features, max_depth, ephemeral_func=partial(np.random.uniform, -5.0, 5.0), normal_distribution_parameters=normal_distribution_parameters)
    structure.register_encoders([CountsEncoder(structure, True, 100), LevelWiseCountsEncoder(structure, True, 100), OneHotEncoder(structure)])
    ground_truths = [NumNodesNegComputer(), MathElasticModelComputer(),
                     WeightsSumNegComputer(structure, 100),
                     LevelWiseWeightsSumNegComputer(structure, 100), NodeWiseWeightsSumNegComputer(structure, 100)]
    ground_truths_names = [g.get_name() for g in ground_truths]

    set_random_seed(101)
    folder_name = "tree_data_3"
    amount_of_feedback = 150
    train_size = 500
    validation_size = 300
    test_size = 250
    data_generator = DatasetGenerator(folder_name, structure, train_size, validation_size, test_size, 101)

    data_generator.generate_tree_encodings(True)

    data_generator.generate_ground_truth(ground_truths)

    for e in structure.get_encoding_type_strings():
        for i in range(2):
            data_generator.create_dataset_warm_up_from_encoding_ground_truth(20, e, ground_truths[i], 102)

    set_random_seed(103)

    data_generator.create_dataset_warm_up_from_csv(folder_name+"/FeynmanEquationsWarmUp.csv",
                                                   "feynman", 20)

    data_generator.persist("datasets")
    print("Starting...")
    exp_exec = ExpsExecutor(data_generator, 700, 10)
    df_list = []
    for enc in structure.get_encoding_type_strings():
        for gro in ground_truths_names:
            for unc in [RandomSamplerOnlineFactory(), UncertaintySamplerOnlineFactory()]:
                for war in [None, "feynman", "elastic_model"]:
                    df_list.append(exp_exec.create_dict_experiment_nn_ranking_online(folder_name, enc, gro,
                                                 amount_of_feedback, nn.ReLU(),
                                             nn.Tanh(), [150, 50], device, sampler=unc,
                                             warmup=war))

    df = PlotGenerator.merge_dictionaries_of_list(df_list)
    PicklePersist.compress_pickle(folder_name+"/dict_res", pd.DataFrame(df))
