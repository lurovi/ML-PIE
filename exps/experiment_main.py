import random

import genepro
from genepro import node_impl as node_impl
from genepro.util import tree_from_prefix_repr
from deeplearn.comparator.OneOutputNeuronsSigmoidComparatorFactory import OneOutputNeuronsSigmoidComparatorFactory
from deeplearn.comparator.TwoOutputNeuronsSoftmaxComparatorFactory import TwoOutputNeuronsSoftmaxComparatorFactory

from deeplearn.model.MLPNet import *
from deeplearn.trainer.StandardBatchTrainer import StandardBatchTrainer

from deeplearn.trainer.TwoPointsCompareTrainer import TwoPointsCompareTrainer
from exps.DatasetGenerator import DatasetGenerator

from exps.ExpsExecutor import ExpsExecutor
from util.PicklePersist import PicklePersist
from util.TorchSeedWorker import TorchSeedWorker

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    # Setting random seed to allow scientific reproducibility
    seed = 2001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator_data_loader = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator_data_loader.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

