import random
from typing import List, Callable

import pandas as pd
from genepro.node_impl import Constant

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from genepro.node import Node
from genepro import node_impl
from genepro.util import tree_from_prefix_repr

import numpy as np

from nsgp.structure.TreeStructure import TreeStructure
from util.PicklePersist import PicklePersist
from util.TreeEncoder import TreeEncoder


class DatasetGenerator:
    def __init__(self, folder_path: str,
                 operators: List[Node],
                 n_features: int,
                 max_depth: int,
                 weights: List[List[float]] = None,
                 constants: List[Constant] = None,
                 ephemeral_func: Callable = None):
        self.__structure = TreeStructure(operators=operators, n_features=n_features, max_depth=max_depth,
                                                weights=weights, constants=constants, ephemeral_func=ephemeral_func)
        self.__folder_path = folder_path
        self.__encodings = ["counts", "level_wise_counts", "one_hot"]

    def generate_trees(self, train_size: int, validation_size: int, test_size: int):
        train = [self.__structure.generate_tree() for _ in range(train_size)]
        val = [self.__structure.generate_tree() for _ in range(validation_size)]
        test = [self.__structure.generate_tree() for _ in range(test_size)]
        PicklePersist.compress_pickle(self.__folder_path + "/original_trees", {"training": train,
                                                                               "validation": val,
                                                                               "test": test})
        PicklePersist.compress_pickle(self.__folder_path + "/tree_structure", self.__structure)
        for enc in self.__encodings:
            scaler = self.__structure.generate_scaler_on_encoding(enc)
            PicklePersist.compress_pickle(self.__folder_path + "/"+enc+"_scaler",
                                          scaler)
            PicklePersist.compress_pickle(self.__folder_path + "/"+enc+"_train_trees", [for t in train])

    @staticmethod
    def create_datasets(operators: List[Node], n_features: int, max_depth: int, folder: str) -> None:
        size = len(operators) + n_features + 1
        n_layers = max_depth + 1
        number_of_distr = 10
        weights = [
            [
                [-abs(np.random.normal(0, 1)) for _ in range(size)]
                for _ in range(n_layers)
            ]
            for _ in range(number_of_distr)
        ]
        structure = TreeGrammarStructure(operators, n_features, max_depth, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0))
        train = [structure.generate_tree() for _ in range(100000)]
        val = [structure.generate_tree() for _ in range(4000)]
        test = [structure.generate_tree() for _ in range(1000)]
        PicklePersist.compress_pickle(folder+"/train_trees", train)
        PicklePersist.compress_pickle(folder+"/validation_trees", val)
        PicklePersist.compress_pickle(folder+"/test_trees", test)
        train, val, test = None, None, None
        PicklePersist.compress_pickle(folder+"/counts_scaler", structure.generate_scaler_on_encoding("level_wise_counts"))
        train = PicklePersist.decompress_pickle(folder+"/train_trees.pbz2")
        val = PicklePersist.decompress_pickle(folder+"/validation_trees.pbz2")
        test = PicklePersist.decompress_pickle(folder+"/test_trees.pbz2")
        scaler = PicklePersist.decompress_pickle(folder+"/counts_scaler.pbz2")

        # TARGET: NUMBER OF NODES

        X_train, y_train = TreeEncoder.create_dataset_level_wise_counts_as_input_number_of_nodes_as_target(train,
                                                                                                           structure,
                                                                                                           scaler)
        X_dev, y_dev = TreeEncoder.create_dataset_level_wise_counts_as_input_number_of_nodes_as_target(val, structure,
                                                                                                       scaler)
        X_test, y_test = TreeEncoder.create_dataset_level_wise_counts_as_input_number_of_nodes_as_target(test,
                                                                                                         structure,
                                                                                                         scaler)
        PicklePersist.compress_pickle(folder+"/counts_number_of_nodes_trees",
                                      {"training": NumericalData(X_train, y_train),
                                       "validation": NumericalData(X_dev, y_dev),
                                       "test": NumericalData(X_test, y_test)})

        X_train, y_train = TreeEncoder.create_dataset_onehot_as_input_number_of_nodes_as_target(train,
                                                                                                structure)
        X_dev, y_dev = TreeEncoder.create_dataset_onehot_as_input_number_of_nodes_as_target(val,
                                                                                            structure)
        X_test, y_test = TreeEncoder.create_dataset_onehot_as_input_number_of_nodes_as_target(test,
                                                                                              structure)
        PicklePersist.compress_pickle(folder+"/onehot_number_of_nodes_trees",
                                      {"training": NumericalData(X_train, y_train),
                                       "validation": NumericalData(X_dev, y_dev),
                                       "test": NumericalData(X_test, y_test)})

        # TARGET: WEIGHTS SUM

        for i in range(number_of_distr):
            curr_weights = weights[i]
            structure.set_weights(curr_weights)

            X_train, y_train = TreeEncoder.create_dataset_level_wise_counts_as_input_weights_sum_as_target(train,
                                                                                                           structure,
                                                                                                           scaler)
            X_dev, y_dev = TreeEncoder.create_dataset_level_wise_counts_as_input_weights_sum_as_target(val,
                                                                                                       structure,
                                                                                                       scaler)
            X_test, y_test = TreeEncoder.create_dataset_level_wise_counts_as_input_weights_sum_as_target(test,
                                                                                                         structure,
                                                                                                         scaler)
            PicklePersist.compress_pickle(folder+"/counts_weights_sum_trees_" + str(i + 1),
                                          {"training": NumericalData(X_train, y_train),
                                           "validation": NumericalData(X_dev, y_dev),
                                           "test": NumericalData(X_test, y_test)})

            X_train, y_train = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(train,
                                                                                                structure)
            X_dev, y_dev = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(val,
                                                                                            structure)
            X_test, y_test = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(test,
                                                                                              structure)
            PicklePersist.compress_pickle(folder+"/onehot_weights_sum_trees_" + str(i + 1),
                                          {"training": NumericalData(X_train, y_train),
                                           "validation": NumericalData(X_dev, y_dev),
                                           "test": NumericalData(X_test, y_test)})

    @staticmethod
    def create_datasets_custom_weights(operators: List[Node], n_features: int, max_depth: int, folder: str, weights: List[List[float]]) -> None:
        size = len(operators) + n_features + 1
        n_layers = max_depth + 1
        number_of_distr = 10
        index = 11

        structure = TreeGrammarStructure(operators, n_features, max_depth, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0))

        train = PicklePersist.decompress_pickle(folder + "/train_trees.pbz2")
        val = PicklePersist.decompress_pickle(folder + "/validation_trees.pbz2")
        test = PicklePersist.decompress_pickle(folder + "/test_trees.pbz2")
        scaler = PicklePersist.decompress_pickle(folder + "/counts_scaler.pbz2")

        structure.set_weights(weights)

        X_train, y_train = TreeEncoder.create_dataset_level_wise_counts_as_input_weights_sum_as_target(train,
                                                                                                       structure,
                                                                                                       scaler)
        X_dev, y_dev = TreeEncoder.create_dataset_level_wise_counts_as_input_weights_sum_as_target(val,
                                                                                                   structure,
                                                                                                   scaler)
        X_test, y_test = TreeEncoder.create_dataset_level_wise_counts_as_input_weights_sum_as_target(test,
                                                                                                     structure,
                                                                                                     scaler)
        PicklePersist.compress_pickle(folder + "/counts_weights_sum_trees_" + str(index),
                                      {"training": NumericalData(X_train, y_train),
                                       "validation": NumericalData(X_dev, y_dev),
                                       "test": NumericalData(X_test, y_test)})

        X_train, y_train = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(train,
                                                                                            structure)
        X_dev, y_dev = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(val,
                                                                                        structure)
        X_test, y_test = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(test,
                                                                                          structure)
        PicklePersist.compress_pickle(folder + "/onehot_weights_sum_trees_" + str(index),
                                      {"training": NumericalData(X_train, y_train),
                                       "validation": NumericalData(X_dev, y_dev),
                                       "test": NumericalData(X_test, y_test)})

    @staticmethod
    def create_dataset_n_nodes_add_prop_warm_up_pairs(operators: List[Node], n_features: int, max_depth: int, folder: str):
        structure = TreeGrammarStructure(operators, n_features, max_depth,
                                         ephemeral_func=lambda: np.random.uniform(-5.0, 5.0))
        scaler = PicklePersist.decompress_pickle(folder + "/counts_scaler.pbz2")
        train = [structure.generate_tree() for _ in range(5000)]

        # GROUND-TRUTH: NUMBER OF NODES

        X_train, y_train = TreeEncoder.create_dataset_level_wise_counts_as_input_number_of_nodes_as_target(train,
                                                                                                           structure,
                                                                                                           scaler)
        dataset = NumericalData(X_train, y_train)
        X_train, y_train = dataset.get_points_and_labels()
        X_train, y_train, _ = PairSampler.random_sampler(X_train, y_train, [], 20)

        PicklePersist.compress_pickle(folder + "/counts_number_of_nodes_warmup_pairs", NumericalData(X_train, y_train))

        # GROUND_TRUTH: ADDITIONAL PROPERTIES OF COUNTS ENCODING

        X_train, y_train = TreeEncoder.create_dataset_level_wise_counts_as_input_add_prop_as_target(train,
                                                                                                    structure,
                                                                                                    scaler)
        dataset = NumericalData(X_train, y_train)
        X_train, y_train = dataset.get_points_and_labels()
        X_train, y_train, _ = PairSampler.random_sampler(X_train, y_train, [], 20)

        PicklePersist.compress_pickle(folder + "/counts_add_prop_warmup_pairs", NumericalData(X_train, y_train))

    @staticmethod
    def create_dataset_feynman_warm_up(folder: str):
        fey_eq = pd.read_csv("D:/shared_folder/python_projects/ML-PIE/util/feynman/dataset/FeynmanEquations.csv")
        fey_eq_reg = pd.read_csv(
            "D:/shared_folder/python_projects/ML-PIE/util/feynman/dataset/FeynmanEquationsRegularized.csv")
        fey_eq_wu = pd.read_csv(
            "D:/shared_folder/python_projects/ML-PIE/util/feynman/dataset/FeynmanEquationsWarmUp.csv")
        fey_eq_wu.drop("Unnamed: 0", axis=1, inplace=True)
        #fey_eq_wu.drop(15, axis=0, inplace=True)  # this rows contains x_1_0
        #fey_eq_wu.drop(42, axis=0, inplace=True)  # this rows contains x_2_0
        #fey_eq_wu.drop(23, axis=0, inplace=True)  # this rows contains x_3_0
        #fey_eq_wu.drop(55, axis=0, inplace=True)  # this rows contains x_3_1 x_3_2
        #fey_eq_wu.drop(83, axis=0, inplace=True)  # this rows contains x_3_1 x_3_2

        feymann_operators = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                             node_impl.UnaryMinus(), node_impl.Power(), node_impl.Square(), node_impl.Cube(),
                             node_impl.Sqrt(), node_impl.Exp(),
                             node_impl.Log(), node_impl.Sin(),
                             node_impl.Cos()]

        structure_feymann = TreeGrammarStructure(feymann_operators, 7, 5, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0))

        scaler = PicklePersist.decompress_pickle("D:/shared_folder/python_projects/ML-PIE/exps/data_genepro_2/counts_scaler.pbz2")
        X_counts_first, X_counts_second, X_onehot_first, X_onehot_second, y = [], [], [], [], []
        n_pairs = len(fey_eq_wu)
        '''
        heights = []
        operators_symb, features_symb = [], []
        count = 0
        for i in range(n_pairs):
            jump = False
            first_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 0])
            second_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 1])
            if first_formula.get_height() > 5:
                jump = True
            feat = first_formula.retrieve_features_from_tree()
            for f in feat:
                if int(f[2:]) > 6:
                    jump = True
            oper = first_formula.retrieve_operators_from_tree()
            for o in oper:
                if o.startswith("arc") or o == "tanh":
                    jump = True
            if jump:
                continue
            heights.append(first_formula.get_height())
            #heights.append(second_formula.get_height())
            operators_symb.extend(first_formula.retrieve_operators_from_tree())
            features_symb.extend(first_formula.retrieve_features_from_tree())
        print(list(set(operators_symb)))
        print(list(set(features_symb)))
        print(len(heights))
        '''
        train_size = 50

        cc = 0
        for i in range(n_pairs):
            first_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 0])
            second_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 1])
            first_counts = structure_feymann.generate_encoding("level_wise_counts", first_formula)
            second_counts = structure_feymann.generate_encoding("level_wise_counts", second_formula)
            first_onehot = structure_feymann.generate_encoding("one_hot", first_formula)
            second_onehot = structure_feymann.generate_encoding("one_hot", second_formula)
            X_counts_first.append(first_counts)
            X_counts_second.append(second_counts)
            X_onehot_first.append(first_onehot)
            X_onehot_second.append(second_onehot)
        n_pairs -= cc

        X_counts_first = scaler.transform(np.array(X_counts_first))
        X_counts_second = scaler.transform(np.array(X_counts_second))
        X_onehot_first = np.array(X_onehot_first)
        X_onehot_second = np.array(X_onehot_second)

        X_counts_pairs, X_onehot_pairs = [], []
        for i in range(n_pairs):
            first_counts, second_counts = X_counts_first[i], X_counts_second[i]
            first_onehot, second_onehot = X_onehot_first[i], X_onehot_second[i]
            if random.uniform(0.0, 1.0) < 0.50:
                counts_pair = np.concatenate((first_counts, second_counts), axis=None)
                onehot_pair = np.concatenate((first_onehot, second_onehot), axis=None)
                y.append(-1)
            else:
                counts_pair = np.concatenate((second_counts, first_counts), axis=None)
                onehot_pair = np.concatenate((second_onehot, first_onehot), axis=None)
                y.append(1)
            X_counts_pairs.append(counts_pair)
            X_onehot_pairs.append(onehot_pair)
        y = np.array(y)
        X_counts_pairs, X_onehot_pairs = np.array(X_counts_pairs), np.array(X_onehot_pairs)
        y_train, y_test = y[:train_size], y[train_size:]
        X_counts_train, X_counts_test, X_onehot_train, X_onehot_test = X_counts_pairs[:train_size], X_counts_pairs[train_size:], X_onehot_pairs[:train_size], X_onehot_pairs[train_size:]
        PicklePersist.compress_pickle(folder+"/feynman_pairs",
                                      {"counts_training": NumericalData(X_counts_train, y_train),
                                       "counts_test": NumericalData(X_counts_test, y_test),
                                       "onehot_training": NumericalData(X_onehot_train, y_train),
                                       "onehot_test": NumericalData(X_onehot_test, y_test),
                                       "counts": NumericalData(X_counts_pairs, y),
                                       "onehot": NumericalData(X_onehot_pairs, y)})
