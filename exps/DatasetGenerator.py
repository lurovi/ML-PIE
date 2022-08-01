from sklearn.preprocessing import MaxAbsScaler

from deeplearn.neuralnet import TreeData, TreeDataTwoPointsCompare
from gp.tree.HalfHalfGenerator import HalfHalfGenerator
from util.PicklePersist import PicklePersist
from util.TreeEncoder import TreeEncoder


class DatasetGenerator:

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
        PicklePersist.compress_pickle("data/onehot_number_of_nodes_trees", {"training": TreeData(X_train, y_train),
                                                         "validation": TreeData(X_dev, y_dev),
                                                         "test": TreeData(X_test, y_test)})

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
        PicklePersist.compress_pickle("data/counts_number_of_nodes_trees", {"training": TreeData(X_train, y_train, scaler),
                                                         "validation": TreeData(X_dev, y_dev, scaler),
                                                         "test": TreeData(X_test, y_test, scaler)})

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
            PicklePersist.compress_pickle("data/onehot_weights_average_trees_"+str(i+1), {"training": TreeData(X_train, y_train),
                                                             "validation": TreeData(X_dev, y_dev),
                                                             "test": TreeData(X_test, y_test)})

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
            PicklePersist.compress_pickle("data/counts_weights_average_trees_"+str(i+1), {"training": TreeData(X_train, y_train, scaler),
                                                             "validation": TreeData(X_dev, y_dev, scaler),
                                                             "test": TreeData(X_test, y_test, scaler)})

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
                                          {"training": TreeData(X_train, y_train),
                                           "validation": TreeData(X_dev, y_dev),
                                           "test": TreeData(X_test, y_test)})

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
                                          {"training": TreeData(X_train, y_train, scaler),
                                           "validation": TreeData(X_dev, y_dev, scaler),
                                           "test": TreeData(X_test, y_test, scaler)})

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
        PicklePersist.compress_pickle("onehot_hci_score_trees", {"training": TreeData(X_train, y_train),
                                                                 "validation": TreeData(X_dev, y_dev),
                                                                 "test": TreeData(X_test, y_test)})

        X_train, y_train = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(train)
        scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_dev, y_dev = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(val)
        X_test, y_test = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(test)
        PicklePersist.compress_pickle("counts_hci_score_trees", {"training": TreeData(X_train, y_train, scaler),
                                                   "validation": TreeData(X_dev, y_dev, scaler),
                                                   "test": TreeData(X_test, y_test, scaler)})

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
