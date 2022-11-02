import random

import torch

from functools import partial

from exps.groundtruth.MathElasticModelComputer import MathElasticModelComputer
from genepro import node_impl
from nsgp.encoder.CountsEncoder import CountsEncoder
from nsgp.structure.TreeStructure import TreeStructure
from util.PicklePersist import PicklePersist

import numpy as np

# settings
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tree parameters
n_features_boston, n_features_heating = 13, 8
phi = MathElasticModelComputer()
duplicates_elimination_data_boston = np.random.uniform(-5.0, 5.0, size=(5, n_features_boston))
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
duplicates_elimination_data_heating = np.random.uniform(-5.0, 5.0, size=(5, n_features_heating))
internal_nodes = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                  node_impl.Cube(),
                  node_impl.Log(), node_impl.Max()]
normal_distribution_parameters_boston = [(0, 1), (0, 1), (0, 3), (0, 8),
                                         (0, 8),
                                         (0, 30), (0, 15)] + [(0, 0.8)] * n_features_boston + [(0, 0.5)]
normal_distribution_parameters_heating = [(0, 1), (0, 1), (0, 3), (0, 8),
                                          (0, 8),
                                          (0, 30), (0, 15)] + [(0, 0.8)] * n_features_heating + [(0, 0.5)]
structure_boston = TreeStructure(internal_nodes, n_features_boston, 5,
                                 ephemeral_func=partial(np.random.uniform, -5.0, 5.0),
                                 normal_distribution_parameters=normal_distribution_parameters_boston)
structure_heating = TreeStructure(internal_nodes, n_features_heating, 5,
                                  ephemeral_func=partial(np.random.uniform, -5.0, 5.0),
                                  normal_distribution_parameters=normal_distribution_parameters_heating)

print("encoder heating init...")
tree_encoder_heating = CountsEncoder(structure_heating, True, 100)
PicklePersist.compress_pickle(title="heating_counts_encoder", data=tree_encoder_heating)
print("encoder heating ready")

print("encoder boston init...")
tree_encoder_boston = CountsEncoder(structure_boston, True, 100)
PicklePersist.compress_pickle(title="boston_counts_encoder", data=tree_encoder_boston)
print("encoder boston ready")
