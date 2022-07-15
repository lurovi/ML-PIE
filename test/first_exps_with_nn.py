from config.setting import *
from gputil.tree import *
from torchutil.neuralnet import *
from torchutil.encodetree import *


# ========================================================
# MAIN
# ========================================================


def ephe_0():
    return random.random()


def ephe_1():
    return float(random.randint(0, 4))


def sum_f(a, b):
    return a + b


def sub_f(a, b):
    return a - b


def mul_f(a, b):
    return a * b


def max_f(a, b):
    return max(a, b)


def min_f(a, b):
    return min(a, b)


def abs_f(a):
    return abs(a)


def neg(a):
    return -a


def power2(a):
    return a**2


def mulby2(a):
    return a*2.0


def divby2(a):
    return a/2.0


def execute_experiment_nn_regression(title, file_name, activation_func, final_activation_func, hidden_layer_sizes, device, max_epochs=20, batch_size=1000):
    trees = decompress_pickle(file_name)
    training, validation, test = trees["training"], trees["validation"], trees["test"]
    input_layer_size = len(validation[0][0])
    output_layer_size = 1
    trainloader = DataLoader(training, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                             generator=generator_data_loader)
    valloader = DataLoader(validation, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                           generator=generator_data_loader)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes)
    trainer = StandardBatchTrainer(net, device, trainloader, nn.MSELoss(reduction="mean"), optimizer_name="adam",
                                   verbose=True, max_epochs=max_epochs)
    trainer.train()
    print(title, " - R2 Score on Validation Set - ", trainer.evaluate_regressor(valloader))


def execute_experiment_nn_ranking(title, file_name_training, file_name_dataset, train_size, activation_func, final_activation_func, hidden_layer_sizes, device, max_epochs=20, batch_size=1000):
    trees = decompress_pickle(file_name_dataset)
    training = decompress_pickle(file_name_training)["training"]
    validation, test = trees["validation"], trees["test"]
    input_layer_size = len(validation[0][0])
    output_layer_size = 1
    trainloader = DataLoader(Subset(training, list(range(train_size))), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                             generator=generator_data_loader)
    valloader = DataLoader(validation, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                           generator=generator_data_loader)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes)
    trainer = TwoPointsCompareTrainer(net, device, trainloader,
                                   verbose=True, max_epochs=max_epochs)
    trainer.train()
    print(title, " - Spearman Footrule on Validation Set - ", trainer.evaluate_ranking(valloader))


def execute_experiment_nn_ranking_with_warmup(title, file_name_warmup, file_name_training, file_name_validation, train_size, activation_func, final_activation_func, hidden_layer_sizes, device, max_epochs_warmup=20, batch_size_warmup=1000, max_epochs=20, batch_size=1000):
    validation_test = decompress_pickle(file_name_validation)
    validation, test = validation_test["validation"], validation_test["test"]
    warmup = decompress_pickle(file_name_warmup)["training"]
    training = decompress_pickle(file_name_training)["training"]
    input_layer_size = len(validation[0][0])
    output_layer_size = 1
    warmuploader = DataLoader(warmup, batch_size=batch_size_warmup, shuffle=True, worker_init_fn=seed_worker,
                             generator=generator_data_loader)
    trainloader = DataLoader(Subset(training, list(range(train_size))), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                             generator=generator_data_loader)
    valloader = DataLoader(validation, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                           generator=generator_data_loader)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes)
    warmupper = StandardBatchTrainer(net, device, warmuploader, nn.MSELoss(reduction="mean"), optimizer_name="adam",
                                     verbose=True, max_epochs=max_epochs_warmup)
    warmupper.train()
    trainer = TwoPointsCompareTrainer(warmupper.model(), device, trainloader,
                                   verbose=True, max_epochs=max_epochs)
    trainer.train()
    print(title, " - Spearman Footrule on Validation Set - ", trainer.evaluate_ranking(valloader))


if __name__ == '__main__':

    constants_0 = [Constant("five", 5.0), Constant("ten", 10.0)]
    ephemeral_0 = [Ephemeral("epm0", ephe_0), Ephemeral("epm1", ephe_1)]

    terminal_set_0 = TerminalSet([float] * 4, constants_0, ephemeral_0)

    primitives_0 = [Primitive("+", float, [float, float], sum_f),
                    Primitive("-", float, [float, float], sub_f),
                    Primitive("*", float, [float, float], mul_f),
                    Primitive("max", float, [float, float], max_f),
                    Primitive("min", float, [float, float], min_f),
                    Primitive("^2", float, [float], power2),
                    Primitive("/2", float, [float], divby2),
                    ]

    primitive_set_0 = PrimitiveSet(primitives_0, float)

    weights_dict = [{"+": 0.90, "-": 0.70, "*": 0.55, "max": 0.21, "min": 0.21,
            "^2": 0.16, "/2": 0.57,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12},
           {"+": 0.88, "-": 0.68, "*": 0.55, "max": 0.23, "min": 0.23,
            "^2": 0.18, "/2": 0.57,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12},
           {"+": 0.86, "-": 0.66, "*": 0.56, "max": 0.24, "min": 0.24,
            "^2": 0.18, "/2": 0.56,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12},
           {"+": 0.85, "-": 0.65, "*": 0.56, "max": 0.25, "min": 0.25,
            "^2": 0.18, "/2": 0.55,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12},
           {"+": 0.84, "-": 0.64, "*": 0.57, "max": 0.26, "min": 0.26,
            "^2": 0.19, "/2": 0.54,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12},
           {"+": 0.82, "-": 0.62, "*": 0.57, "max": 0.27, "min": 0.27,
            "^2": 0.19, "/2": 0.54,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12},
           {"+": 0.81, "-": 0.61, "*": 0.58, "max": 0.28, "min": 0.28,
            "^2": 0.20, "/2": 0.53,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12},
           {"+": 0.80, "-": 0.60, "*": 0.60, "max": 0.30, "min": 0.30,
            "^2": 0.22, "/2": 0.52,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12}
           ]

    '''
    for i in range(3):
        tr = gen_half_half(primitive_set_0, terminal_set_0, 2, 6)
    print(tr.print_as_tree())
    print("\n")
    print(tr)
    print(tr.compile([3, 2, 4, 5]))
    print(tr.count_primitives())
    print(tr.extract_counting_features_from_tree())
    '''

    #train = [gen_half_half(primitive_set_0, terminal_set_0, 2, 6) for _ in range(400000)]
    #val = [gen_half_half(primitive_set_0, terminal_set_0, 2, 6) for _ in range(100000)]
    #test = [gen_half_half(primitive_set_0, terminal_set_0, 2, 6) for _ in range(60000)]
    #compress_pickle("train_trees", train)
    #compress_pickle("validation_trees", val)
    #compress_pickle("test_trees", test)
    train = decompress_pickle("train_trees.pbz2")
    val = decompress_pickle("validation_trees.pbz2")
    test = decompress_pickle("test_trees.pbz2")

    '''
    X_train, y_train = build_dataset_onehot_as_input_weights_average_as_target(train, weights_dict)
    X_dev, y_dev = build_dataset_onehot_as_input_weights_average_as_target(val, weights_dict)
    X_test, y_test = build_dataset_onehot_as_input_weights_average_as_target(test, weights_dict)
    compress_pickle("onehot_weights_average_trees", {"training": TreeData(X_train, y_train),
                                                        "validation": TreeData(X_dev, y_dev),
                                                        "test": TreeData(X_test, y_test)})

    X_train, y_train = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(train, weights_dict)
    X_dev, y_dev = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(val, weights_dict)
    X_test, y_test = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(test, weights_dict)
    compress_pickle("onehot_hci_score_trees", {"training": TreeData(X_train, y_train),
                                                     "validation": TreeData(X_dev, y_dev),
                                                     "test": TreeData(X_test, y_test)})
    
    X_train, y_train = build_dataset_counts_as_input_weights_average_as_target(train, weights_dict)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_dev, y_dev = build_dataset_counts_as_input_weights_average_as_target(val, weights_dict)
    X_test, y_test = build_dataset_counts_as_input_weights_average_as_target(test, weights_dict)
    compress_pickle("counts_weights_average_trees", {"training": TreeData(X_train, y_train, scaler),
                                               "validation": TreeData(X_dev, y_dev, scaler),
                                               "test": TreeData(X_test, y_test, scaler)})

    X_train, y_train = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(train, weights_dict)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_dev, y_dev = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(val, weights_dict)
    X_test, y_test = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(test, weights_dict)
    compress_pickle("counts_hci_score_trees", {"training": TreeData(X_train, y_train, scaler),
                                                "validation": TreeData(X_dev, y_dev, scaler),
                                                "test": TreeData(X_test, y_test, scaler)})
    '''

    '''
    # execute_experiment_nn_regression("Onehot Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [400, 220, 80, 25])",
    #                                 "onehot_weights_average_trees.pbz2",
    #                                 nn.ReLU(), nn.Sigmoid(), [400, 220, 80, 25], device, max_epochs=14, batch_size=1000)
    execute_experiment_nn_regression(
        "Onehot Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25])",
        "onehot_weights_average_trees.pbz2",
        nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device, max_epochs=14, batch_size=1000)

    # execute_experiment_nn_regression("Counts Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [400, 220, 80, 25])",
    #                                 "counts_weights_average_trees.pbz2",
    #                                 nn.ReLU(), nn.Sigmoid(), [100, 60, 30, 15], device, max_epochs=14, batch_size=1000)
    # execute_experiment_nn_regression("Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25])",
    #                                 "counts_weights_average_trees.pbz2",
    #                                 nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device, max_epochs=14, batch_size=1000)
    '''

    '''
    compress_pickle("onehot_weights_average_trees_twopointscomparelarge",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("onehot_weights_average_trees.pbz2")["training"], 120000)})
    compress_pickle("onehot_weights_average_trees_twopointscomparesmall",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("onehot_weights_average_trees.pbz2")["training"], 45000)})

    compress_pickle("counts_weights_average_trees_twopointscomparelarge",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("counts_weights_average_trees.pbz2")["training"], 120000)})
    compress_pickle("counts_weights_average_trees_twopointscomparesmall",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("counts_weights_average_trees.pbz2")["training"], 45000)})
    '''

    '''
    execute_experiment_nn_ranking("Onehot Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Large Training Data.",
                                  "onehot_weights_average_trees_twopointscomparelarge.pbz2",
                                  "onehot_weights_average_trees.pbz2", 120000, nn.ReLU(), nn.Identity(),
                                  [400, 220, 80, 25], device, max_epochs=14, batch_size=1000)
    

    #execute_experiment_nn_ranking("Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Large Training Data.",
    #                              "counts_weights_average_trees_twopointscomparelarge.pbz2",
    #                              "counts_weights_average_trees.pbz2", 120000, nn.ReLU(), nn.Identity(),
    #                              [400, 220, 80, 25], device, max_epochs=14, batch_size=1000)


    execute_experiment_nn_ranking_with_warmup("Onehot Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Large Training Data. Warm Up: Weights average.",
                                              "onehot_weights_average_trees.pbz2",
                                              "onehot_weights_average_trees_twopointscomparelarge.pbz2",
                                              "onehot_weights_average_trees.pbz2",
                                              120000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
                                              max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
                                              batch_size=1000)

    execute_experiment_nn_ranking_with_warmup(
        "Onehot Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Large Training Data. Warm Up: HCI score.",
        "onehot_hci_score_trees.pbz2",
        "onehot_weights_average_trees_twopointscomparelarge.pbz2",
        "onehot_weights_average_trees.pbz2",
        120000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
        max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
        batch_size=1000)

    #execute_experiment_nn_ranking_with_warmup(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Large Training Data. Warm Up: Weights average.",
    #    "counts_weights_average_trees.pbz2",
    #    "counts_weights_average_trees_twopointscomparelarge.pbz2",
    #    "counts_weights_average_trees.pbz2",
    #    120000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
    #    max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
    #    batch_size=1000)


    #execute_experiment_nn_ranking_with_warmup(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Large Training Data. Warm Up: HCI score.",
    #    "counts_hci_score_trees.pbz2",
    #    "counts_weights_average_trees_twopointscomparelarge.pbz2",
    #    "counts_weights_average_trees.pbz2",
    #    120000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
    #    max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
    #    batch_size=1000)

    '''

    execute_experiment_nn_ranking(
        "Onehot Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data.",
        "onehot_weights_average_trees_twopointscomparesmall.pbz2",
        "onehot_weights_average_trees.pbz2", 10000, nn.ReLU(), nn.Identity(),
        [400, 220, 80, 25], device, max_epochs=14, batch_size=1)

    #execute_experiment_nn_ranking(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data.",
    #    "counts_weights_average_trees_twopointscomparesmall.pbz2",
    #    "counts_weights_average_trees.pbz2", 10000, nn.ReLU(), nn.Identity(),
    #    [400, 220, 80, 25], device, max_epochs=14, batch_size=1)

    execute_experiment_nn_ranking_with_warmup(
        "Onehot Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data. Warm Up: Weights average.",
        "onehot_weights_average_trees.pbz2",
        "onehot_weights_average_trees_twopointscomparesmall.pbz2",
        "onehot_weights_average_trees.pbz2",
        10000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
        max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
        batch_size=1)

    execute_experiment_nn_ranking_with_warmup(
        "Onehot Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data. Warm Up: HCI score.",
        "onehot_hci_score_trees.pbz2",
        "onehot_weights_average_trees_twopointscomparesmall.pbz2",
        "onehot_weights_average_trees.pbz2",
        10000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
        max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
        batch_size=1)

    #execute_experiment_nn_ranking_with_warmup(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data. Warm Up: Weights average.",
    #    "counts_weights_average_trees.pbz2",
    #    "counts_weights_average_trees_twopointscomparesmall.pbz2",
    #    "counts_weights_average_trees.pbz2",
    #    10000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
    #    max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
    #    batch_size=1)

    #execute_experiment_nn_ranking_with_warmup(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data. Warm Up: HCI score.",
    #    "counts_hci_score_trees.pbz2",
    #    "counts_weights_average_trees_twopointscomparesmall.pbz2",
    #    "counts_weights_average_trees.pbz2",
    #    10000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
    #    max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
    #    batch_size=1)

