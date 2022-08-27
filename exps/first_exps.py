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


def execute_experiment_nn_regression(title, file_name, activation_func, final_activation_func, hidden_layer_sizes,
                                     device, max_epochs=20, batch_size=1000, optimizer_name="adam", momentum=0.9):
    trees = PicklePersist.decompress_pickle(file_name)
    training, validation, test = trees["training"], trees["validation"], trees["test"]
    input_layer_size = len(validation[0][0])
    output_layer_size = 1
    trainloader = DataLoader(training, batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker,
                             generator=generator_data_loader)
    valloader = DataLoader(validation, batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker,
                           generator=generator_data_loader)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes,
                 dropout_prob=0.25)
    trainer = StandardBatchTrainer(net, device, trainloader, nn.MSELoss(reduction="mean"),
                                   optimizer_name=optimizer_name, momentum=momentum,
                                   verbose=True, is_classification_task=False, max_epochs=max_epochs)
    trainer.train()
    print(title, " - R2 Score on Validation Set - ", trainer.evaluate_regressor(valloader))


def execute_experiment_nn_ranking(title, file_name_training, file_name_dataset, train_size, activation_func,
                                  final_activation_func, hidden_layer_sizes, device, max_epochs=20, batch_size=1000,
                                  optimizer_name="adam", momentum=0.9):
    accs, ftrs = [], []
    for _ in range(10):
        trees = PicklePersist.decompress_pickle(file_name_dataset)
        training = PicklePersist.decompress_pickle(file_name_training)["training"]
        validation, test = trees["validation"], trees["test"]
        input_layer_size = len(validation[0][0])
        output_layer_size = 1
        trainloader = DataLoader(Subset(training, list(range(train_size))), batch_size=batch_size, shuffle=True,
                                 worker_init_fn=TorchSeedWorker.seed_worker,
                                 generator=generator_data_loader)
        valloader = DataLoader(Subset(validation, list(range(500))), batch_size=batch_size, shuffle=True,
                               worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
        # valloader = DataLoader(validation.remove_ground_truth_duplicates(), batch_size=batch_size, shuffle=True,worker_init_fn=TorchSeedWorker.seed_worker,generator=generator_data_loader)
        # trainloader_original = DataLoader(Subset(training.to_simple_torch_dataset(), list(range(train_size * 2))), batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
        net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes,
                     dropout_prob=0.25)
        trainer = TwoPointsCompareTrainer(net, device, trainloader, verbose=True, max_epochs=max_epochs)
        trainer.train()
        # eval_val = trainer.evaluate_ranking(valloader)
        # eval_train = trainer.evaluate_ranking(trainloader_original)
        # print(title, " - Spearman Footrule on Training Set - ", eval_train)
        # print(title, " - Spearman Footrule on Validation Set - ", eval_val)
        # print("Accuracy: ", trainer.evaluate_classifier(valloader))
        accs.append(trainer.evaluate_classifier(valloader))
        ftrs.append(trainer.evaluate_ranking(valloader))
    return sum(accs) / float(len(accs)), sum(ftrs) / float(len(ftrs))


def execute_experiment_nn_ranking_with_warmup(title, file_name_warmup, file_name_training, file_name_validation,
                                              train_size, activation_func, final_activation_func, hidden_layer_sizes,
                                              device, max_epochs_warmup=20, batch_size_warmup=1000, max_epochs=20,
                                              batch_size=1000, optimizer_name="adam", momentum=0.9):
    validation_test = PicklePersist.decompress_pickle(file_name_validation)
    validation, test = validation_test["validation"], validation_test["test"]
    warmup = PicklePersist.decompress_pickle(file_name_warmup)["training"]
    training = PicklePersist.decompress_pickle(file_name_training)["training"]
    input_layer_size = len(validation[0][0])
    output_layer_size = 1
    warmuploader = DataLoader(warmup, batch_size=batch_size_warmup, shuffle=True,
                              worker_init_fn=TorchSeedWorker.seed_worker,
                              generator=generator_data_loader)
    trainloader = DataLoader(Subset(training, list(range(train_size))), batch_size=batch_size, shuffle=True,
                             worker_init_fn=TorchSeedWorker.seed_worker,
                             generator=generator_data_loader)
    valloader = DataLoader(validation, batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker,
                           generator=generator_data_loader)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes,
                 dropout_prob=0.25)
    warmupper = StandardBatchTrainer(net, device, warmuploader, nn.MSELoss(reduction="mean"),
                                     optimizer_name=optimizer_name, momentum=momentum,
                                     verbose=True, is_classification_task=False, max_epochs=max_epochs_warmup)
    warmupper.train()
    trainer = TwoPointsCompareTrainer(warmupper.model(), device, trainloader,
                                      verbose=True, max_epochs=max_epochs)
    trainer.train()
    print(title, " - Spearman Footrule on Validation Set - ", trainer.evaluate_ranking(valloader))


def plot_multiple_experiments_nn_ranking(title, file_name_training, file_name_dataset, max_train_size, activation_func,
                                         final_activation_func, hidden_layer_sizes, device, max_epochs=1, batch_size=1,
                                         optimizer_name="adam", momentum=0.9):
    num_iters = np.arange(50, max_train_size + 1, 50)
    df = {"Training Size": [], "Footrule": [], "Partition": []}
    for train_size in num_iters:
        eval_train, eval_val = execute_experiment_nn_ranking(title, file_name_training, file_name_dataset, train_size,
                                                             activation_func, final_activation_func, hidden_layer_sizes,
                                                             device, max_epochs=max_epochs, batch_size=batch_size,
                                                             optimizer_name=optimizer_name, momentum=momentum)
        df["Training Size"].extend([train_size] * 2)
        df["Footrule"].append(eval_train)
        df["Footrule"].append(eval_val)
        df["Partition"].extend(["Training Set", "Validation Set"])
    plot = sns.lineplot(data=df, x="Training Size", y="Footrule", hue="Partition")
    plt.show()
    return plot


def plot_multiple_experiments_nn_ranking_double_input(title, file_name_training, file_name_dataset, max_train_size,
                                                      activation_func, final_activation_func, hidden_layer_sizes,
                                                      output_layer_size, device, is_classification_task, comparator_fn,
                                                      loss_fn, max_epochs=20, batch_size=1000, optimizer_name="adam",
                                                      momentum=0.9):
    num_iters = np.arange(50, max_train_size + 1, 50)
    df = {"Training Size": [], "Footrule": [], "Partition": []}
    for train_size in num_iters:
        eval_train, eval_val = execute_experiment_nn_ranking_double_input(title, file_name_training, file_name_dataset,
                                                                          train_size, activation_func,
                                                                          final_activation_func, hidden_layer_sizes,
                                                                          output_layer_size, device,
                                                                          is_classification_task, comparator_fn,
                                                                          loss_fn, max_epochs, batch_size,
                                                                          optimizer_name, momentum)
        df["Training Size"].extend([train_size] * 2)
        df["Footrule"].append(eval_train)
        df["Footrule"].append(eval_val)
        df["Partition"].extend(["Training Set", "Validation Set"])
    plot = sns.lineplot(data=df, x="Training Size", y="Footrule", hue="Partition")
    plt.show()
    return plot


def execute_experiment_rf_ranking_double_input(title, file_name_dataset, seed):
    trees = PicklePersist.decompress_pickle(file_name_dataset)
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
    eval = evaluate_ml_ranking_with_spearman_footrule(trees["validation"][0][:500], trees["validation"][1][:500],
                                                      estimator)
    print(title, " - Spearman Footrule on Validation Set - ", eval)


def execute_experiment_regression_with_pwis_and_rf(title, trees_dataset, file_name_dataset, seed,
                                                   collect_feedback=False):
    trees = PicklePersist.decompress_pickle(file_name_dataset)
    original_trees = PicklePersist.decompress_pickle(trees_dataset)
    model = ensemble.RandomForestRegressor(random_state=seed)
    scaler = MaxAbsScaler()
    pipe = Pipeline([('scaler', scaler), ('model', model)])
    space = {"model__n_estimators": [50, 100, 150, 200, 300, 500, 700, 1000, 2000],
             "model__criterion": ["squared_error", "absolute_error", "poisson"],
             "model__max_depth": [20, 30, 40, 50, 70, 100, 200, 500],
             "model__max_features": [None, "sqrt", "log2"]
             }
    estimator = MLEstimator(pipe, space, scoring="r2", random_state=seed,
                            n_splits=5, n_repeats=3, n_jobs=-1,
                            randomized_search=True, n_iter=20)
    training_set, training_labels, validation_set, validation_labels = trees["training"][0][:1000], trees["training"][
                                                                                                        1][:1000], \
                                                                       trees["validation"][0][:1000], \
                                                                       trees["validation"][1][:1000]
    feedback_collector = RawTerminalFeedbackCollector(original_trees[:1000], training_set, training_labels, 20)
    training_set, training_labels = feedback_collector.collect_feedback(20)
    estimator.train(training_set, training_labels, verbose=True)

    pred = estimator.estimate(validation_set)
    print(r2_score(validation_labels, pred))


if __name__ == '__main__':
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

    # twopointscompareloss = two_points_compare_loss
    mseloss = nn.MSELoss(reduction="mean")
    crossentropyloss = nn.CrossEntropyLoss()

    softmaxcomparator = TwoOutputNeuronsSoftmaxComparatorFactory()
    sigmoidcomparator = OneOutputNeuronsSigmoidComparatorFactory()

    constants_0 = [Constant("five", 5.0), Constant("ten", 10.0)]
    ephemeral_0 = [Ephemeral("epm0", SimpleFunctions.ephe_0), Ephemeral("epm1", SimpleFunctions.ephe_1)]

    terminal_set_0 = TerminalSet([float] * 4, constants_0, ephemeral_0)

    primitives_0 = [Primitive("+", float, [float, float], SimpleFunctions.sum),
                    Primitive("-", float, [float, float], SimpleFunctions.sub),
                    Primitive("*", float, [float, float], SimpleFunctions.mul),
                    Primitive("max", float, [float, float], SimpleFunctions.max),
                    Primitive("min", float, [float, float], SimpleFunctions.min),
                    Primitive("^2", float, [float], SimpleFunctions.power2),
                    Primitive("/2", float, [float], SimpleFunctions.divby2),
                    Primitive("cos", float, [float], SimpleFunctions.cos),
                    Primitive("sin", float, [float], SimpleFunctions.sin)
                    ]

    operators = [genepro.node_impl.Plus(), genepro.node_impl.Minus(), genepro.node_impl.Times(),
                 genepro.node_impl.Max(), genepro.node_impl.Min(),
                 genepro.node_impl.Square(), genepro.node_impl.Exp(),
                 genepro.node_impl.Cos(), genepro.node_impl.Sin(), genepro.node_impl.UnaryMinus(),
                 ]

    primitive_set_0 = PrimitiveSet(primitives_0, float)

    weights_dict_avg_1 = [{"+": 0.9754, "-": 0.7993, "*": 0.5946, "max": 0.2116, "min": 0.2116,
                           "^2": 0.4342, "/2": 0.7341}] * 6

    weights_dict_avg_2 = [{"+": 0.4764, "-": 0.6923, "*": 0.9126, "max": 0.4513, "min": 0.4513,
                           "^2": 0.1215, "/2": 0.9842}] * 6

    weights_dict_avg_3 = [{"+": 0.0754, "-": 0.0993, "*": 0.1976, "max": 0.9034, "min": 0.9034,
                           "^2": 0.8772, "/2": 0.5611}] * 6

    weights_dict_avg_4 = [{"+": 0.5354, "-": 0.5493, "*": 0.0486, "max": 0.6416, "min": 0.6416,
                           "^2": 0.6032, "/2": 0.9941}] * 6

    weights_avg_dict_list = [weights_dict_avg_1, weights_dict_avg_2, weights_dict_avg_3, weights_dict_avg_4]

    weights_dict_sum_1 = [{"+": 0.6354, "-": 0.6093, "*": 0.2046, "max": -0.4116, "min": -0.4116,
                           "^2": -0.2342, "/2": 0.7341, "cos": 0.1221, "sin": 0.1221}] * 6

    weights_dict_sum_2 = [{"+": 0.2264, "-": 0.4923, "*": 0.9126, "max": -0.2513, "min": -0.2513,
                           "^2": -0.6215, "/2": 0.6842}] * 6

    weights_dict_sum_3 = [{"+": -0.5754, "-": -0.5993, "*": -0.4276, "max": 0.7034, "min": 0.7034,
                           "^2": 0.7772, "/2": 0.3611}] * 6

    weights_dict_sum_4 = [{"+": 0.2354, "-": 0.2493, "*": -0.6486, "max": 0.3426, "min": 0.3426,
                           "^2": 0.3011, "/2": 0.5941}] * 6

    weights_sum_dict_list = [weights_dict_sum_1, weights_dict_sum_2, weights_dict_sum_3, weights_dict_sum_4]

    '''
    for i in range(4):
        tr = HalfHalfGenerator(primitive_set_0, terminal_set_0, 2, 6).generate_tree()
        for _ in range(3):
            tr_1 = HalfHalfGenerator(primitive_set_0, terminal_set_0, 2, 6).generate_tree()

    print(tr.print_as_tree())
    print(tr)
    print(tr.find_all_sub_chains())
    print(PrimitiveTree.weight_primitives_ranking([["+", "-"], ["/2"], ["*"], ["^2"], ["max", "min"]]))
    print(tr.compute_weighted_sub_chains_average([["+", "-"], ["/2"], ["*"], ["^2"], ["max", "min"]]))
    print(tr.compute_internal_nodes_weights_average([["+", "-"], ["/2"], ["*"], ["^2"], ["max", "min"]]))
    print(tr.compute_property_and_weights_based_interpretability_score([["+", "-"], ["/2"], ["*"], ["^2"], ["max", "min"]]))
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
    
    for _ in range(3):
        tr = HalfHalfGenerator(primitive_set_0, terminal_set_0, 2, 6).generate_tree()
    print(tr.print_as_tree())
    print(TreeEncoder.one_hot_tree(tr))
    print(TreeEncoder.build_dataset_onehot_as_input_weights_sum_as_target([tr], weights_dict_sum_1))
    '''

    # DatasetGenerator.create_datasets(operators, 4, 5)

    # ExpsExecutor.perform_execution_2(device)

    '''
    df = ExpsExecutor.merge_dictionaries_of_list([
        ExpsExecutor.create_dict_experiment_nn_ranking_online("",
                                                       "data_genepro/counts_weights_sum_trees_1.pbz2",
                                                        500, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=False),
        ExpsExecutor.create_dict_experiment_nn_ranking_online("",
                                                       "data_genepro/counts_weights_sum_trees_1.pbz2",
                                                       500, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=True),
        ExpsExecutor.create_dict_experiment_nn_ranking_online("",
                                                       "data_genepro/onehot_weights_sum_trees_1.pbz2",
                                                       500, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=False),
        ExpsExecutor.create_dict_experiment_nn_ranking_online("",
                                                       "data_genepro/onehot_weights_sum_trees_1.pbz2",
                                                       500, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=True)

    ])
    PicklePersist.compress_pickle("data_genepro/plot_train_size_500", df)

    ExpsExecutor.plot_line(df, "Training size", "Footrule", "Representation", "Sampling")
    '''

    ##################################

    # print(ExpsExecutor.perform_experiment_accuracy_feynman_pairs(device))

    ##################################

    # DatasetGenerator.generate_datasets_rand(terminal_set_0, primitive_set_0)

    # plot_random_ranking(device, DataLoader(decompress_pickle("onehot_number_of_nodes_trees.pbz2")["validation"].remove_ground_truth_duplicates(), batch_size=1, shuffle=True))

    '''
    print(ExpsExecutor.execute_experiment_nn_ranking("",
                                                     "data/counts_number_of_nodes_trees_twopointscompare.pbz2",
                                                     "data/counts_number_of_nodes_trees.pbz2",
                                                     200, nn.ReLU(), nn.Tanh(), [220, 140, 80, 26], device, 1, 1))
    '''

    #print(ExpsExecutor.execute_experiment_nn_ranking_online("",
    #                                                        "data/onehot_number_of_nodes_trees_twopointscompare.pbz2",
    #                                                        "data/onehot_number_of_nodes_trees.pbz2",
    #                                                        200, nn.ReLU(), nn.Tanh(), [220, 140, 80, 26], device))

    #ExpsExecutor.example_execution_2(device, online=True)

    '''
    df = ExpsExecutor.merge_dictionaries_of_list([
        ExpsExecutor.plot_experiment_nn_ranking_online("",
                                                       "data/counts_weights_sum_trees_twopointscompare_1.pbz2",
                                                       "data/counts_weights_sum_trees_1.pbz2",
                                                        200, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=False),
        ExpsExecutor.plot_experiment_nn_ranking_online("",
                                                       "data/counts_weights_sum_trees_twopointscompare_1.pbz2",
                                                       "data/counts_weights_sum_trees_1.pbz2",
                                                       200, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=True),
        ExpsExecutor.plot_experiment_nn_ranking_online("",
                                                       "data/onehot_weights_sum_trees_twopointscompare_1.pbz2",
                                                       "data/onehot_weights_sum_trees_1.pbz2",
                                                       200, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=False),
        ExpsExecutor.plot_experiment_nn_ranking_online("",
                                                       "data/onehot_weights_sum_trees_twopointscompare_1.pbz2",
                                                       "data/onehot_weights_sum_trees_1.pbz2",
                                                       200, nn.ReLU(), nn.Identity(), [220, 140, 80, 26],
                                                       device, uncertainty=True)

    ])
    PicklePersist.compress_pickle("data/plot_train_size", df)

    ExpsExecutor.plot_line(df, "Training size", "Footrule", "Representation", "Sampling")
    '''

    #########################################

    # execute_experiment_regression_with_pwis_and_rf("Regression PWIS", "train_trees.pbz2", "onehot_pwis_trees.pbz2", seed)

    '''
    X_train, y_train = build_dataset_onehot_as_input_weights_average_as_target(train, weights_dict)
    X_train, y_train = build_numpy_dataset_twopointscompare(X_train, y_train, 120000, binary_label=True)
    X_dev, y_dev = build_dataset_onehot_as_input_weights_average_as_target(val, weights_dict)
    X_test, y_test = build_dataset_onehot_as_input_weights_average_as_target(test, weights_dict)
    compress_pickle("onehot_weights_average_trees_twopointscomparebinary_numpy",
                    {"training": (X_train, y_train), "validation": (X_dev, y_dev), "test": (X_test, y_test)})
    '''

    # execute_experiment_rf_ranking_double_input("Onehot Tree (Random Forest)",
    #                                            "onehot_weights_average_trees_twopointscomparebinary_numpy.pbz2",
    #                                            seed)

    #########################################

    '''
    print(ExpsExecutor.execute_experiment_nn_ranking_double_input(
        "Counts Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [220, 140, 80, 26]). Large Training Data.",
        "data/onehot_weights_sum_trees_twopointscomparebinary_1.pbz2",
        "data/onehot_weights_sum_trees_1.pbz2", 200, nn.ReLU(), nn.Identity(),
        hidden_layer_sizes=[220, 140, 80, 26], output_layer_size=2,
        device=device, is_classification_task=True,
        comparator_factory=TwoOutputNeuronsSoftmaxComparatorFactory(), loss_fn=crossentropyloss,
        max_epochs=1, batch_size=1))
    '''

    '''
    print(ExpsExecutor.execute_experiment_nn_ranking_double_input_online(
       "Counts Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [220, 140, 80, 26]). Large Training Data.",
       "data/counts_weights_sum_trees_twopointscomparebinary_1.pbz2",
       "data/counts_weights_sum_trees_1.pbz2", 200, nn.ReLU(), nn.Sigmoid(),
        hidden_layer_sizes=[220, 140, 80, 26], output_layer_size=1,
        device=device, is_classification_task=False,
        comparator_factory=OneOutputNeuronsSigmoidComparatorFactory(), loss_fn=mseloss))
    '''

    # plot = plot_multiple_experiments_nn_ranking(
    #    "Counts Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [140, 80, 26]).",
    #    "counts_number_of_nodes_trees_twopointscompare.pbz2",
    #    "counts_number_of_nodes_trees.pbz2", 5000, nn.ReLU(), nn.Identity(),
    #    hidden_layer_sizes=[210, 140, 80, 26],
    #    device=device, max_epochs=1, batch_size=1)

    # plot = plot_multiple_experiments_nn_ranking_double_input(
    #    "Counts Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [140, 80, 26]).",
    #    "counts_weights_average_trees_twopointscomparebinary.pbz2",
    #    "counts_weights_average_trees.pbz2", 5000, nn.ReLU(), nn.Sigmoid(),
    #    hidden_layer_sizes=[140, 80, 26], output_layer_size=1,
    #    device=device, is_classification_task=False,
    #    comparator_fn=sigmoidcomparator, loss_fn=mseloss, max_epochs=100, batch_size=50)

    # execute_experiment_nn_ranking(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [220, 140, 80, 26]).",
    #    "counts_number_of_nodes_trees_twopointscompare.pbz2",
    #    "counts_number_of_nodes_trees.pbz2", 200, nn.ReLU(), nn.Identity(),
    #    [220, 140, 80, 26], device, max_epochs=1, batch_size=1
    # )

    # execute_experiment_nn_ranking_with_warmup(
    #   "Onehot Tree (Activation: ReLU, Final Activation: Sigmoid, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data. Warm Up: HCI score.",
    #   "onehot_hci_score_trees.pbz2",
    #   "onehot_weights_average_trees_twopointscomparesmall.pbz2",
    #   "onehot_weights_average_trees.pbz2",
    #   6000, nn.ReLU(), nn.Sigmoid(), [400, 220, 80, 25], device,
    #   max_epochs_warmup=8, batch_size_warmup=1000, max_epochs=10,
    #   batch_size=1)

    # execute_experiment_nn_ranking_with_warmup(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data. Warm Up: Weights average.",
    #    "counts_weights_average_trees.pbz2",
    #    "counts_weights_average_trees_twopointscomparesmall.pbz2",
    #    "counts_weights_average_trees.pbz2",
    #    10000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
    #    max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
    #    batch_size=1)

    # execute_experiment_nn_ranking_with_warmup(
    #    "Counts Tree (Activation: ReLU, Final Activation: Identity, Hidden Layer Sizes: [400, 220, 80, 25]). Small Training Data. Warm Up: HCI score.",
    #    "counts_hci_score_trees.pbz2",
    #    "counts_weights_average_trees_twopointscomparesmall.pbz2",
    #    "counts_weights_average_trees.pbz2",
    #    10000, nn.ReLU(), nn.Identity(), [400, 220, 80, 25], device,
    #    max_epochs_warmup=6, batch_size_warmup=1000, max_epochs=14,
    #    batch_size=1)
