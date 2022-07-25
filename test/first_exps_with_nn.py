from sklearn import ensemble

from deeplearn.mlmodel import MLEstimator, evaluate_ml_ranking_with_spearman_footrule, \
    build_numpy_dataset_twopointscompare
from util.setting import *
from deeplearn.neuralnet import *
from gp.encodetree import *
import seaborn as sns
import matplotlib.pyplot as plt


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


def execute_experiment_nn_regression(title, file_name, activation_func, final_activation_func, hidden_layer_sizes, device, max_epochs=20, batch_size=1000, optimizer_name="adam", momentum=0.9):
    trees = decompress_pickle(file_name)
    training, validation, test = trees["training"], trees["validation"], trees["test"]
    input_layer_size = len(validation[0][0])
    output_layer_size = 1
    trainloader = DataLoader(training, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                             generator=generator_data_loader)
    valloader = DataLoader(validation, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                           generator=generator_data_loader)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes, dropout_prob=0.25)
    trainer = StandardBatchTrainer(net, device, trainloader, nn.MSELoss(reduction="mean"), optimizer_name=optimizer_name, momentum=momentum,
                                   verbose=True, is_classification_task=False, max_epochs=max_epochs)
    trainer.train()
    print(title, " - R2 Score on Validation Set - ", trainer.evaluate_regressor(valloader))


def execute_experiment_nn_ranking(title, file_name_training, file_name_dataset, train_size, activation_func, final_activation_func, hidden_layer_sizes, device, max_epochs=20, batch_size=1000, optimizer_name="adam", momentum=0.9):
    trees = decompress_pickle(file_name_dataset)
    training = decompress_pickle(file_name_training)["training"]
    validation, test = trees["validation"], trees["test"]
    input_layer_size = len(validation[0][0])
    output_layer_size = 1
    trainloader = DataLoader(Subset(training, list(range(train_size))), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                             generator=generator_data_loader)
    valloader = DataLoader(Subset(validation, list(range(100))), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                           generator=generator_data_loader)
    trainloader_original = DataLoader(Subset(training.to_simple_torch_dataset(), list(range(train_size * 2))),
                                      batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                                      generator=generator_data_loader)
    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes, dropout_prob=0.25)
    trainer = TwoPointsCompareTrainer(net, device, trainloader,
                                   verbose=True, max_epochs=max_epochs)
    trainer.train()
    eval_val = trainer.evaluate_ranking(valloader)
    eval_train = trainer.evaluate_ranking(trainloader_original)
    print(title, " - Spearman Footrule on Training Set - ", eval_train)
    print(title, " - Spearman Footrule on Validation Set - ", eval_val)
    return eval_train, eval_val


def execute_experiment_nn_ranking_with_warmup(title, file_name_warmup, file_name_training, file_name_validation, train_size, activation_func, final_activation_func, hidden_layer_sizes, device, max_epochs_warmup=20, batch_size_warmup=1000, max_epochs=20, batch_size=1000, optimizer_name="adam", momentum=0.9):
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

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes, dropout_prob=0.25)
    warmupper = StandardBatchTrainer(net, device, warmuploader, nn.MSELoss(reduction="mean"), optimizer_name=optimizer_name, momentum=momentum,
                                     verbose=True, is_classification_task=False, max_epochs=max_epochs_warmup)
    warmupper.train()
    trainer = TwoPointsCompareTrainer(warmupper.model(), device, trainloader,
                                   verbose=True, max_epochs=max_epochs)
    trainer.train()
    print(title, " - Spearman Footrule on Validation Set - ", trainer.evaluate_ranking(valloader))


def execute_experiment_nn_ranking_double_input(title, file_name_training, file_name_dataset, train_size, activation_func, final_activation_func, hidden_layer_sizes, output_layer_size, device, is_classification_task, comparator_fn, loss_fn, max_epochs=20, batch_size=1000, optimizer_name="adam", momentum=0.9):
    trees = decompress_pickle(file_name_dataset)
    training = decompress_pickle(file_name_training)["training"]
    validation, test = trees["validation"], trees["test"]
    input_layer_size = len(training[0][0])
    trainloader = DataLoader(Subset(training, list(range(train_size))), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                             generator=generator_data_loader)
    valloader = DataLoader(Subset(validation, list(range(100))), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                           generator=generator_data_loader)
    trainloader_original = DataLoader(Subset(training.to_simple_torch_dataset(), list(range(train_size*2))), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                             generator=generator_data_loader)
    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes, dropout_prob=0.25)
    trainer = TwoPointsCompareDoubleInputTrainer(net, device, trainloader, comparator_fn=comparator_fn, loss_fn=loss_fn,
                                                 optimizer_name=optimizer_name, momentum=momentum,
                                                 verbose=True, is_classification_task=is_classification_task, max_epochs=max_epochs)
    trainer.train()
    eval_val = trainer.evaluate_ranking(valloader)
    eval_train = trainer.evaluate_ranking(trainloader_original)
    print(title, " - Spearman Footrule on Training Set - ", eval_train)
    print(title, " - Spearman Footrule on Validation Set - ", eval_val)
    return eval_train, eval_val


def plot_multiple_experiments_nn_ranking(title, file_name_training, file_name_dataset, max_train_size, activation_func, final_activation_func, hidden_layer_sizes, device, max_epochs=1, batch_size=1, optimizer_name="adam", momentum=0.9):
    num_iters = np.arange(50, max_train_size + 1, 50)
    df = {"Training Size": [], "Footrule": [], "Partition": []}
    for train_size in num_iters:
        eval_train, eval_val = execute_experiment_nn_ranking(title, file_name_training, file_name_dataset, train_size, activation_func, final_activation_func, hidden_layer_sizes, device, max_epochs=max_epochs, batch_size=batch_size, optimizer_name=optimizer_name, momentum=momentum)
        df["Training Size"].extend([train_size] * 2)
        df["Footrule"].append(eval_train)
        df["Footrule"].append(eval_val)
        df["Partition"].extend(["Training Set", "Validation Set"])
    plot = sns.lineplot(data=df, x="Training Size", y="Footrule", hue="Partition")
    plt.show()
    return plot


def plot_multiple_experiments_nn_ranking_double_input(title, file_name_training, file_name_dataset, max_train_size, activation_func, final_activation_func, hidden_layer_sizes, output_layer_size, device, is_classification_task, comparator_fn, loss_fn, max_epochs=20, batch_size=1000, optimizer_name="adam", momentum=0.9):
    num_iters = np.arange(50, max_train_size+1, 50)
    df = {"Training Size": [], "Footrule": [], "Partition": []}
    for train_size in num_iters:
        eval_train, eval_val = execute_experiment_nn_ranking_double_input(title, file_name_training, file_name_dataset, train_size, activation_func, final_activation_func, hidden_layer_sizes, output_layer_size, device, is_classification_task, comparator_fn, loss_fn, max_epochs, batch_size, optimizer_name, momentum)
        df["Training Size"].extend([train_size]*2)
        df["Footrule"].append(eval_train)
        df["Footrule"].append(eval_val)
        df["Partition"].extend(["Training Set", "Validation Set"])
    plot = sns.lineplot(data=df, x="Training Size", y="Footrule", hue="Partition")
    plt.show()
    return plot


def execute_experiment_rf_ranking_double_input(title, file_name_dataset, seed):
    trees = decompress_pickle(file_name_dataset)
    model = ensemble.RandomForestClassifier(random_state=seed)
    scaler = MaxAbsScaler()
    pipe = Pipeline([('scaler', scaler), ('model', model)])
    space = {"model__n_estimators": [50, 100, 150, 200, 300, 500, 700, 1000, 2000],
             "model__criterion": ["gini", "entropy"],
             "model__max_depth": [20, 30, 40, 50, 70, 100, 200, 500],
             "model__max_features": ["auto", "sqrt", "log2"]
             }
    estimator = MLEstimator(pipe, space, scoring="accuracy", random_state=seed,
                            n_splits=5, n_repeats=3, n_jobs=-1,
                            randomized_search=True, n_iter=20)
    estimator.train(trees["training"][0][:3000], trees["training"][1][:3000], verbose=True)
    eval = evaluate_ml_ranking_with_spearman_footrule(trees["validation"][0][:500], trees["validation"][1][:500], estimator)
    print(title, " - Spearman Footrule on Validation Set - ", eval)


if __name__ == '__main__':

    twopointscompareloss = two_points_compare_loss
    mseloss = nn.MSELoss(reduction="mean")
    crossentropyloss = nn.CrossEntropyLoss()

    tanhcomparator = neuralnet_one_output_neurons_tanh_comparator
    sigmoidcomparator = neuralnet_one_output_neurons_sigmoid_comparator
    softmaxcomparator = neuralnet_two_output_neurons_softmax_comparator
    simplecomparator = neuralnet_two_output_neurons_comparator

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

    weights_dict = [{"+": 97.54, "-": 79.93, "*": 59.46, "max": 21.16, "min": 21.16,
            "^2": 43.42, "/2": 73.41,
            "x0": 82.4, "x1": 82.4, "x2": 82.4, "x3": 82.4,
            "c0": 82.4, "c1": 82.4,
            "e0": 82.4, "e1": 82.4},
            {"+": 97.54, "-": 79.93, "*": 59.46, "max": 21.16, "min": 21.16,
             "^2": 43.42, "/2": 73.41,
             "x0": 82.4, "x1": 82.4, "x2": 82.4, "x3": 82.4,
             "c0": 82.4, "c1": 82.4,
             "e0": 82.4, "e1": 82.4},
            {"+": 97.54, "-": 79.93, "*": 59.46, "max": 21.16, "min": 21.16,
             "^2": 43.42, "/2": 73.41,
             "x0": 82.4, "x1": 82.4, "x2": 82.4, "x3": 82.4,
             "c0": 82.4, "c1": 82.4,
             "e0": 82.4, "e1": 82.4},
            {"+": 97.54, "-": 79.93, "*": 59.46, "max": 21.16, "min": 21.16,
             "^2": 43.42, "/2": 73.41,
             "x0": 82.4, "x1": 82.4, "x2": 82.4, "x3": 82.4,
             "c0": 82.4, "c1": 82.4,
             "e0": 82.4, "e1": 82.4},
            {"+": 97.54, "-": 79.93, "*": 59.46, "max": 21.16, "min": 21.16,
             "^2": 43.42, "/2": 73.41,
             "x0": 82.4, "x1": 82.4, "x2": 82.4, "x3": 82.4,
             "c0": 82.4, "c1": 82.4,
             "e0": 82.4, "e1": 82.4},
            {"+": 97.54, "-": 79.93, "*": 59.46, "max": 21.16, "min": 21.16,
             "^2": 43.42, "/2": 73.41,
             "x0": 82.4, "x1": 82.4, "x2": 82.4, "x3": 82.4,
             "c0": 82.4, "c1": 82.4,
             "e0": 82.4, "e1": 82.4}
           ]

    '''
    for i in range(4):
        tr = gen_half_half(primitive_set_0, terminal_set_0, 2, 6)
        for _ in range(3):
            tr_1 = gen_half_half(primitive_set_0, terminal_set_0, 2, 6)

    print(tr.print_as_tree())
    print(tr)
    print(tr_1.print_as_tree())
    print(tr_1)
    print()
    #print(tr.compile([3, 2, 4, 5]))
    #print(tr.count_primitives())
    #print(tr.extract_counting_features_from_tree())
    #print(tr.replace_subtree(tr_1, 2, 1).print_as_tree())
    #print(tr.replace_subtree(tr_1, 2, 1).print_as_text())
    print(ShrinkMutation().mute(tr).print_as_tree())
    print(ShrinkMutation().mute(tr_1).print_as_tree())
    print(UniformMutation().mute(tr).print_as_tree())
    print(UniformMutation().mute(tr_1).print_as_tree())
    lll = OnePointCrossover().cross([tr, tr_1])
    print(lll[0].print_as_tree())
    print(lll[1].print_as_tree())
    '''

    #train = [gen_half_half(primitive_set_0, terminal_set_0, 2, 6) for _ in range(400000)]
    #val = [gen_half_half(primitive_set_0, terminal_set_0, 2, 6) for _ in range(100000)]
    #test = [gen_half_half(primitive_set_0, terminal_set_0, 2, 6) for _ in range(60000)]
    #compress_pickle("train_trees", train)
    #compress_pickle("validation_trees", val)
    #compress_pickle("test_trees", test)
    #train = decompress_pickle("train_trees.pbz2")
    #val = decompress_pickle("validation_trees.pbz2")
    #test = decompress_pickle("test_trees.pbz2")


    '''
    X_train, y_train = build_dataset_onehot_as_input_weights_average_as_target(train, weights_dict)
    X_dev, y_dev = build_dataset_onehot_as_input_weights_average_as_target(val, weights_dict)
    X_test, y_test = build_dataset_onehot_as_input_weights_average_as_target(test, weights_dict)
    compress_pickle("onehot_weights_average_trees", {"training": TreeData(X_train, y_train),
                                                     "validation": TreeData(X_dev, y_dev),
                                                     "test": TreeData(X_test, y_test)})

    X_train, y_train = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(train)
    X_dev, y_dev = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(val)
    X_test, y_test = build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(test)
    compress_pickle("onehot_hci_score_trees", {"training": TreeData(X_train, y_train),
                                                     "validation": TreeData(X_dev, y_dev),
                                                     "test": TreeData(X_test, y_test)})

    compress_pickle("onehot_weights_average_trees_twopointscompare",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("onehot_weights_average_trees.pbz2")["training"], 120000)})
    
    compress_pickle("onehot_weights_average_trees_twopointscomparebinary",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("onehot_weights_average_trees.pbz2")["training"], 120000, True)})

    X_train, y_train = build_dataset_counts_as_input_weights_average_as_target(train, weights_dict)
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X_dev, y_dev = build_dataset_counts_as_input_weights_average_as_target(val, weights_dict)
    X_test, y_test = build_dataset_counts_as_input_weights_average_as_target(test, weights_dict)
    compress_pickle("counts_weights_average_trees", {"training": TreeData(X_train, y_train, scaler),
                                                     "validation": TreeData(X_dev, y_dev, scaler),
                                                     "test": TreeData(X_test, y_test, scaler)})

    X_train, y_train = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(train)
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X_dev, y_dev = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(val)
    X_test, y_test = build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(test)
    compress_pickle("counts_hci_score_trees", {"training": TreeData(X_train, y_train, scaler),
                                                     "validation": TreeData(X_dev, y_dev, scaler),
                                                     "test": TreeData(X_test, y_test, scaler)})

    compress_pickle("counts_weights_average_trees_twopointscompare",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("counts_weights_average_trees.pbz2")["training"], 120000)})

    compress_pickle("counts_weights_average_trees_twopointscomparebinary",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("counts_weights_average_trees.pbz2")["training"], 120000, True)})
    '''

    '''
    compress_pickle("counts_number_of_nodes_trees_twopointscompare",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("counts_number_of_nodes_trees.pbz2")["training"], 30000)})

    compress_pickle("counts_number_of_nodes_trees_twopointscomparebinary",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("counts_number_of_nodes_trees.pbz2")["training"], 30000, True)})

    compress_pickle("counts_weights_average_trees_twopointscompare",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("counts_weights_average_trees.pbz2")["training"], 30000)})

    compress_pickle("counts_weights_average_trees_twopointscomparebinary",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("counts_weights_average_trees.pbz2")["training"], 30000, True)})

    compress_pickle("onehot_number_of_nodes_trees_twopointscompare",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("onehot_number_of_nodes_trees.pbz2")["training"], 30000)})

    compress_pickle("onehot_number_of_nodes_trees_twopointscomparebinary",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("onehot_number_of_nodes_trees.pbz2")["training"], 30000, True)})

    compress_pickle("onehot_weights_average_trees_twopointscompare",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("onehot_weights_average_trees.pbz2")["training"], 30000)})

    compress_pickle("onehot_weights_average_trees_twopointscomparebinary",
                    {"training": TreeDataTwoPointsCompare(
                        decompress_pickle("onehot_weights_average_trees.pbz2")["training"], 30000, True)})
    '''

    #########################################

    '''
    X_train, y_train = build_dataset_onehot_as_input_weights_average_as_target(train, weights_dict)
    X_train, y_train = build_numpy_dataset_twopointscompare(X_train, y_train, 120000, binary_label=True)
    X_dev, y_dev = build_dataset_onehot_as_input_weights_average_as_target(val, weights_dict)
    X_test, y_test = build_dataset_onehot_as_input_weights_average_as_target(test, weights_dict)
    compress_pickle("onehot_weights_average_trees_twopointscomparebinary_numpy",
                    {"training": (X_train, y_train), "validation": (X_dev, y_dev), "test": (X_test, y_test)})
    '''

    #execute_experiment_rf_ranking_double_input("Onehot Tree (Random Forest)",
    #                                            "onehot_weights_average_trees_twopointscomparebinary_numpy.pbz2",
    #                                            seed)

    #execute_experiment_nn_ranking_double_input(
    #   "Counts Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [140, 80, 26]). Large Training Data.",
    #   "onehot_number_of_nodes_trees_twopointscomparebinary.pbz2",
    #   "onehot_number_of_nodes_trees.pbz2", 120000, nn.ReLU(), nn.Identity(),
    #    hidden_layer_sizes=[140, 80, 26], output_layer_size=2,
    #    device=device, is_classification_task=True,
    #    comparator_fn=softmaxcomparator, loss_fn=crossentropyloss, max_epochs=100, batch_size=1000)

    #plot = plot_multiple_experiments_nn_ranking(
    #    "Counts Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [140, 80, 26]).",
    #    "counts_number_of_nodes_trees_twopointscompare.pbz2",
    #    "counts_number_of_nodes_trees.pbz2", 5000, nn.ReLU(), nn.Identity(),
    #    hidden_layer_sizes=[210, 140, 80, 26],
    #    device=device, max_epochs=1, batch_size=1)

    #plot = plot_multiple_experiments_nn_ranking_double_input(
    #    "Counts Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [140, 80, 26]).",
    #    "counts_weights_average_trees_twopointscomparebinary.pbz2",
    #    "counts_weights_average_trees.pbz2", 5000, nn.ReLU(), nn.Sigmoid(),
    #    hidden_layer_sizes=[140, 80, 26], output_layer_size=1,
    #    device=device, is_classification_task=False,
    #    comparator_fn=sigmoidcomparator, loss_fn=mseloss, max_epochs=100, batch_size=50)

    #execute_experiment_nn_ranking(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [140, 80, 26]).",
    #    "counts_number_of_nodes_trees_twopointscompare.pbz2",
    #    "counts_number_of_nodes_trees.pbz2", 5000, nn.ReLU(), nn.Sigmoid(),
    #    [500, 360, 220, 140, 80, 26], device, max_epochs=1, batch_size=1
    #    )

    #execute_experiment_nn_ranking_with_warmup(
    #   "Onehot Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data. Warm Up: HCI score.",
    #   "onehot_hci_score_trees.pbz2",
    #   "onehot_weights_average_trees_twopointscomparesmall.pbz2",
    #   "onehot_weights_average_trees.pbz2",
    #   6000, nn.ReLU(), nn.Sigmoid(), [400, 220, 80, 25], device,
    #   max_epochs_warmup=8, batch_size_warmup=1000, max_epochs=10,
    #   batch_size=1)

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

