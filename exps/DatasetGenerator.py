import random
from typing import List

import pandas as pd

from deeplearn.dataset.NumericalData import NumericalData
from genepro.node import Node
from genepro import node_impl
from genepro.util import tree_from_prefix_repr
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import numpy as np
from deeplearn.dataset.TreeData import TreeData
from deeplearn.dataset.TreeDataTwoPointsCompare import TreeDataTwoPointsCompare
from gp.tree.HalfHalfGenerator import HalfHalfGenerator
from util.TreeGrammarStructure import TreeGrammarStructure
from util.PicklePersist import PicklePersist
from util.TreeEncoder import TreeEncoder


class DatasetGenerator:

    #########################################################################################################
    # ===================================== DATA GENERATION WITH genepro ====================================
    #########################################################################################################

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
        PicklePersist.compress_pickle(folder+"/counts_scaler",
                                      TreeEncoder.create_scaler_on_level_wise_counts(structure, MinMaxScaler(),
                                                                                     [structure.generate_tree() for _ in
                                                                                      range(1000000)]))
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
    def create_dataset_feynman_warm_up(folder: str):
        fey_eq = pd.read_csv("D:/shared_folder/python_projects/ML-PIE/util/feynman/dataset/FeynmanEquations.csv")
        fey_eq_reg = pd.read_csv(
            "D:/shared_folder/python_projects/ML-PIE/util/feynman/dataset/FeynmanEquationsRegularized.csv")
        fey_eq_wu = pd.read_csv(
            "D:/shared_folder/python_projects/ML-PIE/util/feynman/dataset/FeynmanEquationsWarmUp.csv")
        fey_eq_wu.drop("Unnamed: 0", axis=1, inplace=True)
        fey_eq_wu.drop(15, axis=0, inplace=True)  # this rows contains x_1_0
        fey_eq_wu.drop(42, axis=0, inplace=True)  # this rows contains x_2_0
        fey_eq_wu.drop(23, axis=0, inplace=True)  # this rows contains x_3_0
        fey_eq_wu.drop(55, axis=0, inplace=True)  # this rows contains x_3_1 x_3_2
        fey_eq_wu.drop(83, axis=0, inplace=True)  # this rows contains x_3_1 x_3_2
        train_size = 60

        feymann_operators = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                 node_impl.Sqrt(), node_impl.Exp(),
                 node_impl.Log(), node_impl.Sin(),
                 node_impl.Cos(), node_impl.Arccos(), node_impl.Arcsin(), node_impl.Tanh(), node_impl.UnaryMinus(),
                 node_impl.Power(), node_impl.Max(), node_impl.Min(), node_impl.Square(),
                 node_impl.Cube()
                 ]

        structure_feymann = TreeGrammarStructure(feymann_operators, 10, 7, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0))
        scaler = PicklePersist.decompress_pickle("D:/shared_folder/python_projects/ML-PIE/exps/data_genepro_2/counts_scaler.pbz2")
        X_counts_first, X_counts_second, X_onehot_first, X_onehot_second, y = [], [], [], [], []
        n_pairs = len(fey_eq_wu)
        '''
        heights = []
        count = 0
        for i in range(n_pairs):
            first_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 0])
            second_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 1])
            if first_formula.get_height() == 8:
                continue
            heights.append(first_formula.get_height())
            heights.append(second_formula.get_height())
            if first_formula.get_height() == 7:
                count += 1
        print(count)
        print(max(heights))
        '''
        cc = 0
        for i in range(n_pairs):
            first_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 0])
            second_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 1])
            if first_formula.get_height() == 8:
                cc += 1
                continue
            first_counts = structure_feymann.generate_level_wise_counts_encoding(first_formula, True)
            second_counts = structure_feymann.generate_level_wise_counts_encoding(second_formula, True)
            first_onehot = structure_feymann.generate_one_hot_encoding(first_formula)
            second_onehot = structure_feymann.generate_one_hot_encoding(second_formula)
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


    #########################################################################################################
    # ===================================== DATA GENERATION WITH custom_gp ==================================
    #########################################################################################################

    @staticmethod
    def generate_datasets_rand(terminal_set_0, primitive_set_0):

        primitive_names = primitive_set_0.primitive_names()
        feature_names = ["x" + str(i) for i in range(terminal_set_0.num_features())]
        constant_names = ["c0"]
        all_names = primitive_names + feature_names + constant_names
        min_depth, max_depth, number_of_distr = 2, 6, 10
        weights_dict = [[{k: -abs(np.random.normal(0, 1)) for k in all_names}] * max_depth for _ in range(number_of_distr)]

        train = [HalfHalfGenerator(primitive_set_0, terminal_set_0, min_depth, max_depth).generate_tree() for _ in range(5000)]
        val = [HalfHalfGenerator(primitive_set_0, terminal_set_0, min_depth, max_depth).generate_tree() for _ in range(2000)]
        test = [HalfHalfGenerator(primitive_set_0, terminal_set_0, min_depth, max_depth).generate_tree() for _ in range(1000)]
        PicklePersist.compress_pickle("data/train_trees", train)
        PicklePersist.compress_pickle("data/validation_trees", val)
        PicklePersist.compress_pickle("data/test_trees", test)
        train = PicklePersist.decompress_pickle("data/train_trees.pbz2")
        val = PicklePersist.decompress_pickle("data/validation_trees.pbz2")
        test = PicklePersist.decompress_pickle("data/test_trees.pbz2")

        # TARGET: NUMBER OF NODES

        X_train, y_train = TreeEncoder.build_dataset_onehot_as_input_number_of_nodes_as_target(train)
        X_dev, y_dev = TreeEncoder.build_dataset_onehot_as_input_number_of_nodes_as_target(val)
        X_test, y_test = TreeEncoder.build_dataset_onehot_as_input_number_of_nodes_as_target(test)
        PicklePersist.compress_pickle("data/onehot_number_of_nodes_trees", {"training": TreeData(train, X_train, y_train),
                                                                            "validation": TreeData(val, X_dev, y_dev),
                                                                            "test": TreeData(test, X_test, y_test)})

        PicklePersist.compress_pickle("data/onehot_number_of_nodes_trees_twopointscompare",
                                      {"training": TreeDataTwoPointsCompare(
                                          PicklePersist.decompress_pickle("data/onehot_number_of_nodes_trees.pbz2")[
                                              "training"], 5000)})

        PicklePersist.compress_pickle("data/onehot_number_of_nodes_trees_twopointscomparebinary",
                                      {"training": TreeDataTwoPointsCompare(
                                          PicklePersist.decompress_pickle("data/onehot_number_of_nodes_trees.pbz2")[
                                              "training"], 5000, True)})

        X_train, y_train = TreeEncoder.build_dataset_counts_as_input_number_of_nodes_as_target(train)
        scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_dev, y_dev = TreeEncoder.build_dataset_counts_as_input_number_of_nodes_as_target(val)
        X_test, y_test = TreeEncoder.build_dataset_counts_as_input_number_of_nodes_as_target(test)
        PicklePersist.compress_pickle("data/counts_number_of_nodes_trees",
                                      {"training": TreeData(train, X_train, y_train, scaler),
                                       "validation": TreeData(val, X_dev, y_dev, scaler),
                                       "test": TreeData(test, X_test, y_test, scaler)})

        PicklePersist.compress_pickle("data/counts_number_of_nodes_trees_twopointscompare",
                                      {"training": TreeDataTwoPointsCompare(
                                          PicklePersist.decompress_pickle("data/counts_number_of_nodes_trees.pbz2")[
                                              "training"], 5000)})

        PicklePersist.compress_pickle("data/counts_number_of_nodes_trees_twopointscomparebinary",
                                      {"training": TreeDataTwoPointsCompare(
                                          PicklePersist.decompress_pickle("data/counts_number_of_nodes_trees.pbz2")[
                                              "training"], 5000, True)})

        # TARGET: WEIGHTS SUM

        for i in range(len(weights_dict)):
            X_train, y_train = TreeEncoder.build_dataset_onehot_as_input_weights_sum_as_target(train,
                                                                                               weights_dict[i])
            X_dev, y_dev = TreeEncoder.build_dataset_onehot_as_input_weights_sum_as_target(val,
                                                                                           weights_dict[i])
            X_test, y_test = TreeEncoder.build_dataset_onehot_as_input_weights_sum_as_target(test,
                                                                                             weights_dict[i])
            PicklePersist.compress_pickle("data/onehot_weights_sum_trees_" + str(i + 1),
                                          {"training": TreeData(train, X_train, y_train),
                                           "validation": TreeData(val, X_dev, y_dev),
                                           "test": TreeData(test, X_test, y_test)})

            PicklePersist.compress_pickle("data/onehot_weights_sum_trees_twopointscompare_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/onehot_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000)})

            PicklePersist.compress_pickle("data/onehot_weights_sum_trees_twopointscomparebinary_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/onehot_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000, True)})

            PicklePersist.compress_pickle("data/onehot_weights_sum_trees_twopointscompare_samenodes_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/onehot_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000, False, "nodes")})

            PicklePersist.compress_pickle("data/onehot_weights_sum_trees_twopointscomparebinary_samenodes_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/onehot_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000, True, "nodes")})

            X_train, y_train = TreeEncoder.build_dataset_counts_as_input_weights_sum_as_target(train,
                                                                                               weights_dict[i])
            scaler = MaxAbsScaler()
            scaler.fit(X_train)
            X_dev, y_dev = TreeEncoder.build_dataset_counts_as_input_weights_sum_as_target(val,
                                                                                           weights_dict[i])
            X_test, y_test = TreeEncoder.build_dataset_counts_as_input_weights_sum_as_target(test,
                                                                                             weights_dict[i])
            PicklePersist.compress_pickle("data/counts_weights_sum_trees_" + str(i + 1),
                                          {"training": TreeData(train, X_train, y_train, scaler),
                                           "validation": TreeData(val, X_dev, y_dev, scaler),
                                           "test": TreeData(test, X_test, y_test, scaler)})

            PicklePersist.compress_pickle("data/counts_weights_sum_trees_twopointscompare_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/counts_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000)})

            PicklePersist.compress_pickle("data/counts_weights_sum_trees_twopointscomparebinary_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/counts_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000, True)})

            PicklePersist.compress_pickle("data/counts_weights_sum_trees_twopointscompare_samenodes_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/counts_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000, False, "nodes")})

            PicklePersist.compress_pickle("data/counts_weights_sum_trees_twopointscomparebinary_samenodes_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/counts_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000, True, "nodes")})

    @staticmethod
    def generate_datasets(terminal_set_0, primitive_set_0, weights_avg_dict_list, weights_sum_dict_list):
        train = [HalfHalfGenerator(primitive_set_0, terminal_set_0, 2, 6).generate_tree() for _ in range(5000)]
        val = [HalfHalfGenerator(primitive_set_0, terminal_set_0, 2, 6).generate_tree() for _ in range(2000)]
        test = [HalfHalfGenerator(primitive_set_0, terminal_set_0, 2, 6).generate_tree() for _ in range(1000)]
        PicklePersist.compress_pickle("data/train_trees", train)
        PicklePersist.compress_pickle("data/validation_trees", val)
        PicklePersist.compress_pickle("data/test_trees", test)
        train = PicklePersist.decompress_pickle("data/train_trees.pbz2")
        val = PicklePersist.decompress_pickle("data/validation_trees.pbz2")
        test = PicklePersist.decompress_pickle("data/test_trees.pbz2")

        # TARGET: NUMBER OF NODES

        X_train, y_train = TreeEncoder.build_dataset_onehot_as_input_number_of_nodes_as_target(train)
        X_dev, y_dev = TreeEncoder.build_dataset_onehot_as_input_number_of_nodes_as_target(val)
        X_test, y_test = TreeEncoder.build_dataset_onehot_as_input_number_of_nodes_as_target(test)
        PicklePersist.compress_pickle("data/onehot_number_of_nodes_trees", {"training": TreeData(train, X_train, y_train),
                                                         "validation": TreeData(val, X_dev, y_dev),
                                                         "test": TreeData(test, X_test, y_test)})

        PicklePersist.compress_pickle("data/onehot_number_of_nodes_trees_twopointscompare",
                                      {"training": TreeDataTwoPointsCompare(
                                          PicklePersist.decompress_pickle("data/onehot_number_of_nodes_trees.pbz2")["training"], 5000)})

        PicklePersist.compress_pickle("data/onehot_number_of_nodes_trees_twopointscomparebinary",
                                      {"training": TreeDataTwoPointsCompare(
                                          PicklePersist.decompress_pickle("data/onehot_number_of_nodes_trees.pbz2")["training"], 5000, True)})

        X_train, y_train = TreeEncoder.build_dataset_counts_as_input_number_of_nodes_as_target(train)
        scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_dev, y_dev = TreeEncoder.build_dataset_counts_as_input_number_of_nodes_as_target(val)
        X_test, y_test = TreeEncoder.build_dataset_counts_as_input_number_of_nodes_as_target(test)
        PicklePersist.compress_pickle("data/counts_number_of_nodes_trees", {"training": TreeData(train, X_train, y_train, scaler),
                                                         "validation": TreeData(val, X_dev, y_dev, scaler),
                                                         "test": TreeData(test, X_test, y_test, scaler)})

        PicklePersist.compress_pickle("data/counts_number_of_nodes_trees_twopointscompare",
                                      {"training": TreeDataTwoPointsCompare(
                                          PicklePersist.decompress_pickle("data/counts_number_of_nodes_trees.pbz2")["training"], 5000)})

        PicklePersist.compress_pickle("data/counts_number_of_nodes_trees_twopointscomparebinary",
                                      {"training": TreeDataTwoPointsCompare(
                                          PicklePersist.decompress_pickle("data/counts_number_of_nodes_trees.pbz2")["training"], 5000, True)})

        # TARGET: WEIGHTS AVG

        for i in range(len(weights_avg_dict_list)):
            X_train, y_train = TreeEncoder.build_dataset_onehot_as_input_weights_average_as_target(train, weights_avg_dict_list[i])
            X_dev, y_dev = TreeEncoder.build_dataset_onehot_as_input_weights_average_as_target(val, weights_avg_dict_list[i])
            X_test, y_test = TreeEncoder.build_dataset_onehot_as_input_weights_average_as_target(test, weights_avg_dict_list[i])
            PicklePersist.compress_pickle("data/onehot_weights_average_trees_"+str(i+1), {"training": TreeData(train, X_train, y_train),
                                                             "validation": TreeData(val, X_dev, y_dev),
                                                             "test": TreeData(test, X_test, y_test)})

            PicklePersist.compress_pickle("data/onehot_weights_average_trees_twopointscompare_"+str(i+1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle("data/onehot_weights_average_trees_"+str(i+1)+".pbz2")["training"], 5000)})

            PicklePersist.compress_pickle("data/onehot_weights_average_trees_twopointscomparebinary_"+str(i+1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle("data/onehot_weights_average_trees_"+str(i+1)+".pbz2")["training"], 5000, True)})

            X_train, y_train = TreeEncoder.build_dataset_counts_as_input_weights_average_as_target(train, weights_avg_dict_list[i])
            scaler = MaxAbsScaler()
            scaler.fit(X_train)
            X_dev, y_dev = TreeEncoder.build_dataset_counts_as_input_weights_average_as_target(val, weights_avg_dict_list[i])
            X_test, y_test = TreeEncoder.build_dataset_counts_as_input_weights_average_as_target(test, weights_avg_dict_list[i])
            PicklePersist.compress_pickle("data/counts_weights_average_trees_"+str(i+1), {"training": TreeData(train, X_train, y_train, scaler),
                                                             "validation": TreeData(val, X_dev, y_dev, scaler),
                                                             "test": TreeData(test, X_test, y_test, scaler)})

            PicklePersist.compress_pickle("data/counts_weights_average_trees_twopointscompare_"+str(i+1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle("data/counts_weights_average_trees_"+str(i+1)+".pbz2")["training"], 5000)})

            PicklePersist.compress_pickle("data/counts_weights_average_trees_twopointscomparebinary_"+str(i+1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle("data/counts_weights_average_trees_"+str(i+1)+".pbz2")["training"], 5000, True)})

        # TARGET: WEIGHTS SUM

        for i in range(len(weights_sum_dict_list)):
            X_train, y_train = TreeEncoder.build_dataset_onehot_as_input_weights_sum_as_target(train, weights_sum_dict_list[i])
            X_dev, y_dev = TreeEncoder.build_dataset_onehot_as_input_weights_sum_as_target(val, weights_sum_dict_list[i])
            X_test, y_test = TreeEncoder.build_dataset_onehot_as_input_weights_sum_as_target(test, weights_sum_dict_list[i])
            PicklePersist.compress_pickle("data/onehot_weights_sum_trees_" + str(i + 1),
                                          {"training": TreeData(train, X_train, y_train),
                                           "validation": TreeData(val, X_dev, y_dev),
                                           "test": TreeData(test, X_test, y_test)})

            PicklePersist.compress_pickle("data/onehot_weights_sum_trees_twopointscompare_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/onehot_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000)})

            PicklePersist.compress_pickle("data/onehot_weights_sum_trees_twopointscomparebinary_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/onehot_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000, True)})

            X_train, y_train = TreeEncoder.build_dataset_counts_as_input_weights_sum_as_target(train, weights_sum_dict_list[i])
            scaler = MaxAbsScaler()
            scaler.fit(X_train)
            X_dev, y_dev = TreeEncoder.build_dataset_counts_as_input_weights_sum_as_target(val, weights_sum_dict_list[i])
            X_test, y_test = TreeEncoder.build_dataset_counts_as_input_weights_sum_as_target(test, weights_sum_dict_list[i])
            PicklePersist.compress_pickle("data/counts_weights_sum_trees_" + str(i + 1),
                                          {"training": TreeData(train, X_train, y_train, scaler),
                                           "validation": TreeData(val, X_dev, y_dev, scaler),
                                           "test": TreeData(test, X_test, y_test, scaler)})

            PicklePersist.compress_pickle("data/counts_weights_sum_trees_twopointscompare_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/counts_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000)})

            PicklePersist.compress_pickle("data/counts_weights_sum_trees_twopointscomparebinary_" + str(i + 1),
                                          {"training": TreeDataTwoPointsCompare(
                                              PicklePersist.decompress_pickle(
                                                  "data/counts_weights_sum_trees_" + str(i + 1) + ".pbz2")[
                                                  "training"], 5000, True)})

        '''
        
        X_train, y_train = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(train)
        X_dev, y_dev = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(val)
        X_test, y_test = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(test)
        PicklePersist.compress_pickle("onehot_hci_score_trees", {"training": TreeData(train, X_train, y_train),
                                                                 "validation": TreeData(val, X_dev, y_dev),
                                                                 "test": TreeData(test, X_test, y_test)})

        X_train, y_train = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(train)
        scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_dev, y_dev = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(val)
        X_test, y_test = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(test)
        PicklePersist.compress_pickle("counts_hci_score_trees", {"training": TreeData(train, X_train, y_train, scaler),
                                                   "validation": TreeData(val, X_dev, y_dev, scaler),
                                                   "test": TreeData(test, X_test, y_test, scaler)})

        X_train, y_train = build_dataset_onehot_as_input_pwis_as_target(train,
                                                                        [["+", "-"], ["/2"], ["*"], ["^2"],
                                                                         ["max", "min"]])
        X_dev, y_dev = build_dataset_onehot_as_input_pwis_as_target(val,
                                                                    [["+", "-"], ["/2"], ["*"], ["^2"], ["max", "min"]])
        X_test, y_test = build_dataset_onehot_as_input_pwis_as_target(test,
                                                                      [["+", "-"], ["/2"], ["*"], ["^2"],
                                                                       ["max", "min"]])
        PicklePersist.compress_pickle("onehot_pwis_trees", {"training": (X_train, y_train),
                                              "validation": (X_dev, y_dev),
                                              "test": (X_test, y_test)})
        '''
