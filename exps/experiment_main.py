import random

from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import genepro
from genepro import node_impl as node_impl
from genepro.util import tree_from_prefix_repr
from deeplearn.comparator.OneOutputNeuronsSigmoidComparatorFactory import OneOutputNeuronsSigmoidComparatorFactory
from deeplearn.comparator.TwoOutputNeuronsSoftmaxComparatorFactory import TwoOutputNeuronsSoftmaxComparatorFactory
from deeplearn.mlmodel import MLEstimator, evaluate_ml_ranking_with_spearman_footrule, \
    RawTerminalFeedbackCollector
from deeplearn.model.MLPNet import *
from deeplearn.trainer.StandardBatchTrainer import StandardBatchTrainer
from deeplearn.trainer.TwoPointsCompareDoubleInputTrainer import TwoPointsCompareDoubleInputTrainer
from deeplearn.trainer.TwoPointsCompareTrainer import TwoPointsCompareTrainer
from exps.DatasetGenerator import DatasetGenerator

from exps.ExpsExecutor import ExpsExecutor
from gp.tree.Constant import Constant
from gp.tree.Ephemeral import Ephemeral
from gp.tree.HalfHalfGenerator import HalfHalfGenerator
from gp.tree.PrimitiveSet import PrimitiveSet
from gp.tree.TerminalSet import TerminalSet
from exps.SimpleFunctions import SimpleFunctions
from util.PicklePersist import PicklePersist
from util.TorchSeedWorker import TorchSeedWorker
from util.TreeEncoder import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', None)

if __name__ == "__main__":
    # Setting random seed to allow scientific reproducibility
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator_data_loader = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator_data_loader.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = [
        [-0.005, -0.005, -1.2, -5.0, -15.0, -20.0, -14.0, -40.0, -40.0, -50.0, -50.0, -50.0, -0.001, -0.50, -10.0,
         -10.0, -3.0, -6.0] + [-0.0001] * 11,
        [-0.005, -0.005, -0.85, -3.0, -10.0, -15.0, -12.0, -35.0, -35.0, -40.0, -40.0, -40.0, -0.001, -0.30, -6.0, -6.0,
         -1.67, -4.0] + [-0.0001] * 11,
        [-0.005, -0.005, -0.50, -2.0, -9.0, -10.0, -8.0, -30.0, -30.0, -35.0, -35.0, -30.0, -0.001, -0.20, -5.0, -5.0,
         -0.50, -3.0] + [-0.0001] * 11,
        [-0.005, -0.005, -0.30, -1.5, -5.0, -7.0, -6.0, -20.0, -20.0, -25.0, -25.0, -20.0, -0.001, -0.10, -1.2, -1.2,
         -0.20, -1.5] + [-0.0001] * 11,
        [-0.005, -0.005, -0.15, -0.85, -2.0, -5.0, -3.0, -10.0, -10.0, -15.0, -15.0, -10.0, -0.001, -0.2, -0.4, -0.4,
         -0.009, -1.0] + [-0.0001] * 11,
        [-50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50] + [-0.0001] * 11,
        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100] + [
            -0.0001] * 11,
        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100] + [
            -0.0001] * 11
    ]

    operators = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                 node_impl.Sqrt(), node_impl.Exp(),
                 node_impl.Log(), node_impl.Sin(),
                 node_impl.Cos(), node_impl.Arccos(), node_impl.Arcsin(), node_impl.Tanh(), node_impl.UnaryMinus(),
                 node_impl.Power(), node_impl.Max(), node_impl.Min(), node_impl.Square(),
                 node_impl.Cube()
                 ]
    n_features = 10
    max_depth = 7
    structure = TreeGrammarStructure(operators, n_features, max_depth)

    ##############

    # DatasetGenerator.create_datasets(operators, n_features, max_depth, "data_genepro_2")
    # DatasetGenerator.create_datasets_custom_weights(operators, n_features, max_depth, "data_genepro_2", weights)
    # DatasetGenerator.create_dataset_feynman_warm_up("data_genepro_2")
    # print(ExpsExecutor.perform_experiment_accuracy_feynman_pairs("data_genepro_2", device))

    ##############

    df = ExpsExecutor.merge_dictionaries_of_list([
        ExpsExecutor.create_dict_experiment_nn_ranking_online("", "data_genepro_2",
                                                       "data_genepro_2/counts_weights_sum_trees_11.pbz2",
                                                        500, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=False, warmup="n_nodes"),
        ExpsExecutor.create_dict_experiment_nn_ranking_online("", "data_genepro_2",
                                                       "data_genepro_2/counts_weights_sum_trees_11.pbz2",
                                                       500, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=True, warmup="n_nodes"),
        ExpsExecutor.create_dict_experiment_nn_ranking_online("", "data_genepro_2",
                                                       "data_genepro_2/counts_weights_sum_trees_11.pbz2",
                                                       500, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=False, warmup=None),
        ExpsExecutor.create_dict_experiment_nn_ranking_online("", "data_genepro_2",
                                                       "data_genepro_2/counts_weights_sum_trees_11.pbz2",
                                                       500, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=True, warmup=None)

    ])
    PicklePersist.compress_pickle("data_genepro_2/plot_train_size_number_of_nodes_warmup_500", df)

    ExpsExecutor.plot_line(df, "Training size", "Footrule", "Warm-up", "Sampling")


    ##############

    #print(NeuralNetEvaluator.evaluate_regression(net_model,
    #                                             DataLoader(
    #                                                 PicklePersist.decompress_pickle(
    #                                                     "data_genepro_2/counts_number_of_nodes_trees.pbz2")[
    #                                                     "validation"],
    #                                                 shuffle=True, batch_size=1), device))
