from typing import List

from deeplearn.dataset.NumericalData import NumericalData
from genepro.node import Node
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from deeplearn.dataset.TreeData import TreeData
from deeplearn.dataset.TreeDataTwoPointsCompare import TreeDataTwoPointsCompare
from gp.tree.HalfHalfGenerator import HalfHalfGenerator
from nsgp.TreeGrammarStructure import TreeGrammarStructure
from util.PicklePersist import PicklePersist
from util.TreeEncoder import TreeEncoder


class DatasetGenerator:

    @staticmethod
    def create_datasets(operators: List[Node], n_features: int, max_depth: int, max_arity: int) -> None:
        size = len(operators) + n_features + 1
        n_layers = max_depth + 1
        number_of_distr = 10
        weights = [ [[-abs(np.random.normal(0, 1)) for _ in range(size)]]*n_layers for _ in range(number_of_distr)]
        structure = TreeGrammarStructure(operators, n_features, max_depth, max_arity)

        train = [structure.generate_tree() for _ in range(100000)]
        val = [structure.generate_tree() for _ in range(4000)]
        test = [structure.generate_tree() for _ in range(1000)]
        PicklePersist.compress_pickle("data_genepro/train_trees", train)
        PicklePersist.compress_pickle("data_genepro/validation_trees", val)
        PicklePersist.compress_pickle("data_genepro/test_trees", test)
        PicklePersist.compress_pickle("data_genepro/counts_scaler", TreeEncoder.create_scaler_on_counts(structure, MaxAbsScaler(), train))
        train = PicklePersist.decompress_pickle("data_genepro/train_trees.pbz2")
        val = PicklePersist.decompress_pickle("data_genepro/validation_trees.pbz2")
        test = PicklePersist.decompress_pickle("data_genepro/test_trees.pbz2")
        scaler = PicklePersist.decompress_pickle("data_genepro/counts_scaler.pbz2")

        # TARGET: NUMBER OF NODES

        X_train, y_train = TreeEncoder.create_dataset_counts_as_input_number_of_nodes_as_target(train, structure,
                                                                                                scaler)
        X_dev, y_dev = TreeEncoder.create_dataset_counts_as_input_number_of_nodes_as_target(val, structure,
                                                                                            scaler)
        X_test, y_test = TreeEncoder.create_dataset_counts_as_input_number_of_nodes_as_target(test, structure,
                                                                                              scaler)
        PicklePersist.compress_pickle("data_genepro/counts_number_of_nodes_trees",
                                      {"training": NumericalData(X_train, y_train),
                                       "validation": NumericalData(X_dev, y_dev),
                                       "test": NumericalData(X_test, y_test)})

        X_train, y_train = TreeEncoder.create_dataset_onehot_as_input_number_of_nodes_as_target(train,
                                                                                                structure)
        X_dev, y_dev = TreeEncoder.create_dataset_onehot_as_input_number_of_nodes_as_target(val,
                                                                                            structure)
        X_test, y_test = TreeEncoder.create_dataset_onehot_as_input_number_of_nodes_as_target(test,
                                                                                              structure)
        PicklePersist.compress_pickle("data_genepro/onehot_number_of_nodes_trees",
                                      {"training": NumericalData(X_train, y_train),
                                       "validation": NumericalData(X_dev, y_dev),
                                       "test": NumericalData(X_test, y_test)})

        # TARGET: WEIGHTS SUM

        for i in range(number_of_distr):
            curr_weights = weights[i]
            structure.set_weights(curr_weights)

            X_train, y_train = TreeEncoder.create_dataset_counts_as_input_weights_sum_as_target(train,
                                                                                                structure, scaler)
            X_dev, y_dev = TreeEncoder.create_dataset_counts_as_input_weights_sum_as_target(val,
                                                                                            structure, scaler)
            X_test, y_test = TreeEncoder.create_dataset_counts_as_input_weights_sum_as_target(test,
                                                                                              structure, scaler)
            PicklePersist.compress_pickle("data_genepro/counts_weights_sum_trees_" + str(i + 1),
                                          {"training": NumericalData(X_train, y_train),
                                           "validation": NumericalData(X_dev, y_dev),
                                           "test": NumericalData(X_test, y_test)})

            X_train, y_train = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(train,
                                                                                                structure)
            X_dev, y_dev = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(val,
                                                                                            structure)
            X_test, y_test = TreeEncoder.create_dataset_onehot_as_input_weights_sum_as_target(test,
                                                                                              structure)
            PicklePersist.compress_pickle("data_genepro/onehot_weights_sum_trees_" + str(i + 1),
                                          {"training": NumericalData(X_train, y_train),
                                           "validation": NumericalData(X_dev, y_dev),
                                           "test": NumericalData(X_test, y_test)})

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
