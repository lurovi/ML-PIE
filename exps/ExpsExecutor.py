import math
from functools import partial
from typing import Dict, List, Any, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, Dataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import multiprocessing
import random

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.dataset.TreeData import TreeData
from deeplearn.model.MLPNet import MLPNet
from deeplearn.model.NeuralNetEvaluator import NeuralNetEvaluator
from deeplearn.trainer.OnlineTwoPointsCompareDoubleInputTrainer import OnlineTwoPointsCompareDoubleInputTrainer
from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer
from deeplearn.trainer.StandardBatchTrainerFactory import StandardBatchTrainerFactory
from deeplearn.trainer.TrainerFactory import TrainerFactory
from deeplearn.trainer.TwoPointsCompareDoubleInputTrainer import TwoPointsCompareDoubleInputTrainer
from deeplearn.trainer.TwoPointsCompareTrainer import TwoPointsCompareTrainer
from deeplearn.trainer.TwoPointsCompareTrainerFactory import TwoPointsCompareTrainerFactory
from util.EvaluationMetrics import EvaluationMetrics
from util.PicklePersist import PicklePersist
from util.Sort import Sort
from util.TorchSeedWorker import TorchSeedWorker


class ExpsExecutor:

    #########################################################################################################
    # ===================================== UTILITIES =======================================================
    #########################################################################################################

    @staticmethod
    def merge_dictionaries_of_list(dicts: List[Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        df = dicts[0]
        for i in range(1, len(dicts)):
            df = {k: df[k] + dicts[i][k] for k in df.keys()}
        return df

    @staticmethod
    def plot_line(df, x, y, hue, style, pgfplot=False):
        sns.set(rc={"figure.figsize": (10, 6)})
        sns.set_style("white")
        df = pd.DataFrame(df)
        ax = sns.lineplot(data=df, x=x, y=y, hue=hue, style=style, estimator=np.mean, ci=90)
        if pgfplot:
            matplotlib.use("pgf")
            matplotlib.rcParams.update(
                {"pgf.texsystem": "pdflatex", 'font.family': 'serif', 'font.size': 11, 'text.usetex': True,
                 'pgf.rcfonts': False, })
            plt.savefig("img/plot.pgf")
        else:
            plt.show()
        return ax

    @staticmethod
    def random_comparator(point_1, point_2, p):  # here point is the ground truth label and p a probability
        if random.random() < p:
            return point_1 < point_2
        else:
            return not (point_1 < point_2)

    @staticmethod
    def plot_random_ranking(device, dataloader):
        df = {"Probability": [], "Footrule": []}
        for p in np.arange(0, 1.1, 0.1):
            df["Probability"].append(p)
            ll = sum([ExpsExecutor.random_spearman(device, dataloader, p) for _ in range(20)]) / 20.0
            df["Footrule"].append(ll)
        plot = sns.lineplot(data=df, x="Probability", y="Footrule")
        plt.show()
        return plot

    @staticmethod
    def random_spearman(device, dataloader, p):
        y_true = []
        points = []
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device).float(), labels.to(device).float().reshape((labels.shape[0], 1))
            for i in range(len(inputs)):
                points.append(inputs[i])
                y_true.append(labels[i][0].item())
        y_true_2 = [x for x in y_true]
        y_true, _ = Sort.heapsort(y_true, lambda x, y: x < y, inplace=False, reverse=False)
        comparator = partial(ExpsExecutor.random_comparator, p=p)
        y_pred, _ = Sort.heapsort(y_true_2, comparator, inplace=False, reverse=False)
        return EvaluationMetrics.spearman_footrule(y_true, y_pred, lambda x, y: x == y)

    #########################################################################################################
    # ===================================== SINGLE INPUT ARCHITECTURES WITH genepro =========================
    #########################################################################################################

    @staticmethod
    def perform_experiment_accuracy_feynman_pairs(folder, device):

        counts_accs, counts_ftrs, onehot_accs, onehot_ftrs = [], [], [], []
        verbose = False

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.use_deterministic_algorithms(True)

        data = PicklePersist.decompress_pickle(folder+"/feynman_pairs.pbz2")
        counts_input_layer_size = len(data["counts_training"][0][0])//2
        onehot_input_layer_size = len(data["onehot_training"][0][0])//2
        print(counts_input_layer_size)
        print(onehot_input_layer_size)
        output_layer_size = 1
        print(data["counts_training"].count_labels())
        print(data["counts_test"].count_labels())

        pool = multiprocessing.Pool(4)
        exec_func = partial(parallel_execution_perform_experiment_accuracy_feynman_pairs,
                            data=data, counts_input_layer_size=counts_input_layer_size, onehot_input_layer_size=onehot_input_layer_size, output_layer_size=output_layer_size, device=device)

        exec_res = pool.map(exec_func, list(range(10)))
        pool.close()
        pool.join()
        for curr_acc_counts, curr_acc_onehot in exec_res:
            counts_accs.append(curr_acc_counts)
            onehot_accs.append(curr_acc_onehot)

        return sum(counts_accs) / float(len(counts_accs)), sum(onehot_accs) / float(len(onehot_accs))



    @staticmethod
    def perform_experiment_nn_ranking_online(title, file_name_dataset, train_size, activation_func,
                                             final_activation_func, hidden_layer_sizes, device, uncertainty=False,
                                             optimizer_name="adam", momentum=0.9):
        accs, ftrs = [], []
        verbose = False

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.use_deterministic_algorithms(True)

        trees = PicklePersist.decompress_pickle(file_name_dataset)
        training, validation, test = trees["training"], trees["validation"], trees["test"]
        input_layer_size = len(validation[0][0])
        output_layer_size = 1
        training = training.subset(list(range(train_size*2+200)))
        validation = validation.subset(list(range(700)))
        X_tr, y_tr = training.get_points_and_labels()
        X_va, y_va = validation.get_points_and_labels()
        pairs_X_va, pairs_y_va = PairSampler.random_sampler_with_replacement(X_va, y_va, 2000)
        valloader = DataLoader(validation, batch_size=1, shuffle=True)
        pairs_valloader = DataLoader(NumericalData(pairs_X_va, pairs_y_va), batch_size=1, shuffle=True)

        pool = multiprocessing.Pool(4)
        exec_func = partial(parallel_execution_perform_experiment_nn_ranking_online, activation_func=activation_func, final_activation_func=final_activation_func, input_layer_size=input_layer_size, output_layer_size=output_layer_size, hidden_layer_sizes=hidden_layer_sizes, train_size=train_size, uncertainty=uncertainty, device=device, X_tr=X_tr, y_tr=y_tr, verbose=verbose, valloader=valloader, pairs_valloader=pairs_valloader)
        exec_res = pool.map(exec_func, list(range(10)))
        pool.close()
        pool.join()
        for curr_acc, curr_ftrs in exec_res:
            accs.append(curr_acc)
            ftrs.append(curr_ftrs)
        print(title)

        return sum(accs) / float(len(accs)), sum(ftrs) / float(len(ftrs))


    @staticmethod
    def perform_execution_2(device):
        activations = {"identity": nn.Identity(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        for target in ["number_of_nodes"]:
            for representation in ["counts", "onehot"]:
                for final_activation in ["identity"]:
                    for uncertainty in [False, True]:
                        print(ExpsExecutor.perform_experiment_nn_ranking_online(
                            target + " " + representation + " " + final_activation + " " + ("uncertainty" if uncertainty else "random"),
                            "data_genepro/" + representation + "_" + target + "_" + "trees" + ".pbz2",
                            200,
                            nn.ReLU(), activations[final_activation],
                            [220, 140, 80, 26], device, uncertainty
                        ))
        for target in ["weights_sum"]:
            for i in range(1, 10 + 1):
                i_str = str(i)
                for representation in ["counts", "onehot"]:
                    for final_activation in ["identity"]:
                        for uncertainty in [False, True]:
                            print(ExpsExecutor.perform_experiment_nn_ranking_online(
                                target + " " + i_str + " " + representation + " " + final_activation + " " + ("uncertainty" if uncertainty else "random"),
                                "data_genepro/" + representation + "_" + target + "_" + "trees" + "_" + i_str + ".pbz2",
                                200,
                                nn.ReLU(), activations[final_activation],
                                [220, 140, 80, 26], device, uncertainty
                            ))

    @staticmethod
    def create_dict_experiment_nn_ranking_online(title, folder: str, file_name_dataset, train_size, activation_func,
                                             final_activation_func, hidden_layer_sizes, device, uncertainty=False,
                                             optimizer_name="adam", momentum=0.9,
                                             warmup=None):
        df = {"Training size": [], "Footrule": [], "Representation": [], "Sampling": [], "Warm-up": []}
        verbose = False

        if "/counts_" in file_name_dataset:
            repr_plot = "Counts"
        elif "/onehot_" in file_name_dataset:
            repr_plot = "Onehot"
        else:
            raise AttributeError(f"Bad representation.")
        if uncertainty:
            sampl_plot = "Uncertainty"
        else:
            sampl_plot = "Random"

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.use_deterministic_algorithms(True)

        trees = PicklePersist.decompress_pickle(file_name_dataset)
        training, validation, test = trees["training"], trees["validation"], trees["test"]
        input_layer_size = len(validation[0][0])
        output_layer_size = 1
        training = training.subset(list(range(train_size*2+200)))
        validation = validation.subset(list(range(700)))
        X_tr, y_tr = training.get_points_and_labels()
        # X_va, y_va = validation.get_points_and_labels()
        # pairs_X_va, pairs_y_va = PairSampler.random_sampler_with_replacement(X_va, y_va, 2000)
        valloader = DataLoader(validation, batch_size=1, shuffle=True)
        # pairs_valloader = DataLoader(NumericalData(pairs_X_va, pairs_y_va), batch_size=1, shuffle=True)

        if warmup is not None:
            warmup_plot = "Warm-up"
            if warmup == "feynman":
                feynam_data = PicklePersist.decompress_pickle(folder + "/feynman_pairs.pbz2")
                if repr_plot == "Counts":
                    warmup_data = feynam_data["counts"]
                elif repr_plot == "Onehot":
                    warmup_data = feynam_data["onehot"]
                pretrainer_factory = TwoPointsCompareTrainerFactory(False, 1)
            elif warmup == "n_nodes":
                if repr_plot == "Counts":
                    warmup_data = PicklePersist.decompress_pickle(folder + "/counts_number_of_nodes_warmup_pairs.pbz2")
                elif repr_plot == "Onehot":
                    warmup_data = PicklePersist.decompress_pickle(folder + "/onehot_number_of_nodes_trees.pbz2")["training"].subset(list(range(80)))
                pretrainer_factory = TwoPointsCompareTrainerFactory(False, 1)
            elif warmup == "add_prop":
                if repr_plot == "Counts":
                    warmup_data = PicklePersist.decompress_pickle(folder + "/counts_add_prop_warmup_pairs.pbz2")
                elif repr_plot == "Onehot":
                    warmup_data = PicklePersist.decompress_pickle(folder + "/onehot_number_of_nodes_trees.pbz2")["training"].subset(list(range(80)))
                pretrainer_factory = TwoPointsCompareTrainerFactory(False, 1)
            else:
                raise AttributeError("Bad warmup parameter.")
        else:
            warmup_plot = "No Warm-up"
            warmup_data = None
            pretrainer_factory = None

        pool = multiprocessing.Pool(1)
        exec_func = partial(parallel_execution_create_dict_experiment_nn_ranking_online,activation_func=activation_func, final_activation_func=final_activation_func, input_layer_size=input_layer_size,
                                                                output_layer_size=output_layer_size, hidden_layer_sizes=hidden_layer_sizes,
                                                                device=device, pretrainer_factory=pretrainer_factory,
                                                                warmup_data=warmup_data, train_size=train_size,
                                                                uncertainty=uncertainty, X_tr=X_tr,
                                                                y_tr=y_tr, verbose=verbose,
                                                                valloader=valloader, repr_plot=repr_plot,
                                                                sampl_plot=sampl_plot, warmup_plot=warmup_plot)
        exec_res = pool.map(exec_func, list(range(10)))
        pool.close()
        pool.join()
        for l in exec_res:
            for curr_train_index, curr_ftrs, curr_repr_plot, curr_sampl_plot, curr_warmup_plot in l:
                df["Training size"].append(curr_train_index)
                df["Footrule"].append(curr_ftrs)
                df["Representation"].append(curr_repr_plot)
                df["Sampling"].append(curr_sampl_plot)
                df["Warm-up"].append(curr_warmup_plot)
        print("Done")
        return df

    #########################################################################################################
    # ===================================== SINGLE INPUT ARCHITECTURES WITH custom_gp =======================
    #########################################################################################################

    @staticmethod
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
            trainloader = Subset(training, list(range(train_size)))
            valloader = DataLoader(Subset(validation, list(range(500))), batch_size=batch_size, shuffle=True)
            # valloader = DataLoader(validation.remove_ground_truth_duplicates(), batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
            # trainloader_original = DataLoader(Subset(training.to_simple_torch_dataset(), list(range(train_size * 2))), batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
            net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                         hidden_layer_sizes, dropout_prob=0.25)
            trainer = TwoPointsCompareTrainer(net, device, trainloader, verbose=True, max_epochs=max_epochs)
            trainer.train()
            # eval_val = trainer.evaluate_ranking(valloader)
            # eval_train = trainer.evaluate_ranking(trainloader_original)
            # print(title, " - Spearman Footrule on Training Set - ", eval_train)
            # print(title, " - Spearman Footrule on Validation Set - ", eval_val)
            # print("Accuracy: ", trainer.evaluate_classifier(valloader))
            accs.append(trainer.evaluate_classifier(valloader)[0])
            ftrs.append(trainer.evaluate_ranking(valloader))
        print(title)
        return sum(accs) / float(len(accs)), sum(ftrs) / float(len(ftrs))

    @staticmethod
    def execute_experiment_nn_ranking_online(title, file_name_training, file_name_dataset, train_size, activation_func,
                                      final_activation_func, hidden_layer_sizes, device,
                                      optimizer_name="adam", momentum=0.9):
        accs, ftrs = [], []
        verbose = False
        for _ in range(10):
            trees = PicklePersist.decompress_pickle(file_name_dataset)
            training = PicklePersist.decompress_pickle(file_name_training)["training"]
            validation, test = trees["validation"], trees["test"]
            input_layer_size = len(validation[0][0])
            output_layer_size = 1
            trainloader = Subset(training, list(range(train_size)))
            valloader = DataLoader(Subset(validation, list(range(500))), batch_size=1, shuffle=True)
            # valloader = DataLoader(validation.remove_ground_truth_duplicates(), batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
            # trainloader_original = DataLoader(Subset(training.to_simple_torch_dataset(), list(range(train_size * 2))), batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
            net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                         hidden_layer_sizes, dropout_prob=0.25)
            trainer = OnlineTwoPointsCompareTrainer(net, device, data=None, verbose=False)
            for idx in range(len(trainloader)):
                trainer.change_data(Subset(trainloader, [idx]))
                loss_epoch_array = trainer.train()
                if verbose and idx == len(trainloader) - 1:
                    print(f"Loss: {loss_epoch_array[0]}")
            # eval_val = trainer.evaluate_ranking(valloader)
            # eval_train = trainer.evaluate_ranking(trainloader_original)
            # print(title, " - Spearman Footrule on Training Set - ", eval_train)
            # print(title, " - Spearman Footrule on Validation Set - ", eval_val)
            # print("Accuracy: ", trainer.evaluate_classifier(valloader))
            accs.append(trainer.evaluate_classifier(valloader)[0])
            ftrs.append(trainer.evaluate_ranking(valloader))
        print(title)
        return sum(accs) / float(len(accs)), sum(ftrs) / float(len(ftrs))

    @staticmethod
    def example_execution_1(device):
        activations = {"identity": nn.Identity(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        for target in ["weights_sum"]:
            for i in range(1, 4+1):
                i_str = str(i)
                for representation in ["counts", "onehot"]:
                    for final_activation in ["identity", "sigmoid", "tanh"]:
                        print(ExpsExecutor.execute_experiment_nn_ranking(
                            target + " " + i_str + " " + representation + " " + final_activation,
                            "data/" + representation + "_" + target + "_" + "trees_twopointscompare" + "_" + i_str + ".pbz2",
                            "data/" + representation + "_" + target + "_" + "trees" + "_" + i_str + ".pbz2", 200,
                            nn.ReLU(), activations[final_activation],
                            [220, 140, 80, 26], device, max_epochs=1, batch_size=1
                        ))

    @staticmethod
    def example_execution_2(device, online=True):
        activations = {"identity": nn.Identity(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        for target in ["weights_sum"]:
            for i in range(1, 10+1):
                i_str = str(i)
                for representation in ["counts", "onehot"]:
                    for final_activation in ["identity"]:
                        for criterion in ["target", "nodes"]:
                            twopointscomparename = "trees_twopointscompare" if criterion == "target" else "trees_twopointscompare_samenodes"

                            if not online:
                                print(ExpsExecutor.execute_experiment_nn_ranking(
                                    target + " " + i_str + " " + representation + " " + final_activation + " " + criterion,
                                    "data/" + representation + "_" + target + "_" + twopointscomparename + "_" + i_str + ".pbz2",
                                    "data/" + representation + "_" + target + "_" + "trees" + "_" + i_str + ".pbz2", 200,
                                    nn.ReLU(), activations[final_activation],
                                    [220, 140, 80, 26], device, max_epochs=1, batch_size=1
                                ))
                            else:
                                print(ExpsExecutor.execute_experiment_nn_ranking_online(
                                    target + " " + i_str + " " + representation + " " + final_activation + " " + criterion,
                                    "data/" + representation + "_" + target + "_" + twopointscomparename + "_" + i_str + ".pbz2",
                                    "data/" + representation + "_" + target + "_" + "trees" + "_" + i_str + ".pbz2",
                                    200,
                                    nn.ReLU(), activations[final_activation],
                                    [220, 140, 80, 26], device
                                ))

    @staticmethod
    def plot_experiment_nn_ranking_online(title, file_name_training, file_name_dataset, train_size, activation_func,
                                             final_activation_func, hidden_layer_sizes, device, uncertainty=False,
                                             optimizer_name="adam", momentum=0.9):
        df = {"Training size": [], "Accuracy": [], "Footrule": [], "Representation": [], "Sampling": []}
        accs, ftrs = [], []
        verbose = True

        trees = PicklePersist.decompress_pickle(file_name_dataset)
        training = PicklePersist.decompress_pickle(file_name_training)["training"]
        train_original, validation, test = trees["training"], trees["validation"], trees["test"]
        input_layer_size = len(validation[0][0])
        output_layer_size = 1
        trainloader = Subset(training, list(range(train_size)))
        valloader = DataLoader(Subset(validation, list(range(500))), batch_size=1, shuffle=True)

        train_samples = []
        train_labels = []
        train_original = Subset(train_original, list(range(700)))
        for i in range(len(train_original)):
            train_samples.append(train_original[i][0].tolist())
            train_labels.append(train_original[i][1].item())
        train_samples = torch.tensor(train_samples, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        train_indexes = list(range(len(train_labels)))

        for curr_seed in range(1, 10 + 1):
            random.seed(curr_seed)
            np.random.seed(curr_seed)
            torch.manual_seed(curr_seed)
            torch.use_deterministic_algorithms(True)

            curr_accs, curr_ftrs = [], []
            net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                         hidden_layer_sizes, dropout_prob=0.25)
            trainer = OnlineTwoPointsCompareTrainer(net, device, data=None, verbose=False)
            already_seen = []
            for idx in range(len(trainloader)):
                if not uncertainty:
                    #trainer.change_data(Subset(trainloader, [idx]))
                    exit_loop = False
                    while not (exit_loop):
                        idx_1 = random.choice(train_indexes)
                        if idx_1 not in already_seen:
                            exit_loop = True
                            already_seen.append(idx_1)
                            first_point, first_label = train_samples[idx_1], train_labels[idx_1].item()
                    exit_loop = False
                    while not(exit_loop):
                        idx_2 = random.choice(train_indexes)
                        if idx_2 != idx_1 and idx_2 not in already_seen:
                            exit_loop = True
                            already_seen.append(idx_2)
                            second_point, second_label = train_samples[idx_2], train_labels[idx_2].item()
                    if first_label >= second_label:
                        curr_feedback = np.array([-1])
                    else:
                        curr_feedback = np.array([1])
                    curr_point = np.array([first_point.tolist() + second_point.tolist()])
                    trainer.change_data(TreeData(None, curr_point, curr_feedback, scaler=None))
                else:
                    _, uncertainty = trainer.predict(train_samples)
                    _, ind_points = Sort.heapsort(uncertainty, lambda x, y: x < y, inplace=False, reverse=True)
                    count = 0
                    i = 0
                    points = []
                    while count < 2 and i < len(ind_points):
                        if ind_points[i] not in already_seen:
                            already_seen.append(ind_points[i])
                            count += 1
                            points.append((train_samples[ind_points[i]], train_labels[ind_points[i]].item()))
                        i += 1
                    first_point, first_label = points[0]
                    second_point, second_label = points[1]
                    if first_label >= second_label:
                        curr_feedback = np.array([-1])
                    else:
                        curr_feedback = np.array([1])
                    curr_point = np.array([first_point.tolist() + second_point.tolist()])
                    trainer.change_data(TreeData(None, curr_point, curr_feedback, scaler=None))
                loss_epoch_array = trainer.train()
                #curr_accs.append(trainer.evaluate_classifier(valloader)[0])
                curr_ftrs.append(trainer.evaluate_ranking(valloader))
                if verbose and idx == len(trainloader) - 1:
                    print(f"Loss: {loss_epoch_array[0]}")
            accs.append(curr_accs)
            ftrs.append(curr_ftrs)
        #accs = np.array(accs)
        ftrs = np.array(ftrs)
        #mean_acc = np.mean(accs, axis=0)
        mean_ftrs = np.mean(ftrs, axis=0)
        for i in range(len(mean_ftrs)):
            df["Training size"].append(i+1)
            #df["Accuracy"].append(mean_acc[i])
            df["Footrule"].append(mean_ftrs[i])
            if "/counts_" in file_name_dataset:
                df["Representation"].append("Counts")
            elif "/onehot_" in file_name_dataset:
                df["Representation"].append("Onehot")
            else:
                raise AttributeError(f"Bad representation.")
            if uncertainty:
                df["Sampling"].append("Uncertainty")
            else:
                df["Sampling"].append("Random")
        return df

    #########################################################################################################
    # ===================================== DOUBLE INPUT ARCHITECTURES WITH custom_gp =======================
    #########################################################################################################

    @staticmethod
    def execute_experiment_nn_ranking_double_input(title, file_name_training, file_name_dataset, train_size,
                                                   activation_func, final_activation_func, hidden_layer_sizes,
                                                   output_layer_size, device, is_classification_task,
                                                   comparator_factory, loss_fn, max_epochs=20, batch_size=1000,
                                                   optimizer_name="adam", momentum=0.9):
        accs, ftrs = [], []
        for curr_seed in range(1,10+1):
            random.seed(curr_seed)
            np.random.seed(curr_seed)
            torch.manual_seed(curr_seed)

            trees = PicklePersist.decompress_pickle(file_name_dataset)
            training = PicklePersist.decompress_pickle(file_name_training)["training"]
            validation, test = trees["validation"], trees["test"]
            input_layer_size = len(training[0][0])
            trainloader = Subset(training, list(range(train_size)))
            valloader = DataLoader(Subset(validation, list(range(500))), batch_size=batch_size, shuffle=True)
            trainloader_original = DataLoader(Subset(training.to_simple_torch_dataset(), list(range(train_size * 2))),
                                              batch_size=batch_size, shuffle=True)
            net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                         hidden_layer_sizes,
                         dropout_prob=0.25)
            trainer = TwoPointsCompareDoubleInputTrainer(net, device, comparator_factory=comparator_factory,
                                                         loss_fn=loss_fn, data=trainloader,
                                                         optimizer_name=optimizer_name, momentum=momentum,
                                                         verbose=True, is_classification_task=is_classification_task,
                                                         max_epochs=max_epochs)
            trainer.train()
            # eval_val = trainer.evaluate_ranking(valloader)
            # eval_train = trainer.evaluate_ranking(trainloader_original)
            # print(title, " - Spearman Footrule on Training Set - ", eval_train)
            # print(title, " - Spearman Footrule on Validation Set - ", eval_val)
            # print("Accuracy:  ", trainer.evaluate_classifier(valloader))
            accs.append(trainer.evaluate_classifier(valloader)[0])
            ftrs.append(trainer.evaluate_ranking(valloader))
        print(title)
        return sum(accs) / float(len(accs)), sum(ftrs) / float(len(ftrs))

    @staticmethod
    def execute_experiment_nn_ranking_double_input_online(title, file_name_training, file_name_dataset, train_size,
                                                   activation_func, final_activation_func, hidden_layer_sizes,
                                                   output_layer_size, device, is_classification_task,
                                                   comparator_factory, loss_fn,
                                                   optimizer_name="adam", momentum=0.9):
        accs, ftrs = [], []
        verbose = False
        for _ in range(10):
            trees = PicklePersist.decompress_pickle(file_name_dataset)
            training = PicklePersist.decompress_pickle(file_name_training)["training"]
            validation, test = trees["validation"], trees["test"]
            input_layer_size = len(training[0][0])
            trainloader = Subset(training, list(range(train_size)))
            valloader = DataLoader(Subset(validation, list(range(500))), batch_size=1, shuffle=True)
            trainloader_original = DataLoader(Subset(training.to_simple_torch_dataset(), list(range(train_size * 2))),
                                              batch_size=1, shuffle=True)
            net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                         hidden_layer_sizes,
                         dropout_prob=0.25)
            trainer = OnlineTwoPointsCompareDoubleInputTrainer(net, device, comparator_factory=comparator_factory,
                                                         loss_fn=loss_fn, data=None,
                                                         optimizer_name=optimizer_name, momentum=momentum,
                                                         verbose=False, is_classification_task=is_classification_task)

            for idx in range(len(trainloader)):
                trainer.change_data(Subset(trainloader, [idx]))
                loss_epoch_array = trainer.train()
                if verbose and idx == len(trainloader) - 1:
                    print(f"Loss: {loss_epoch_array[0]}")
            # eval_val = trainer.evaluate_ranking(valloader)
            # eval_train = trainer.evaluate_ranking(trainloader_original)
            # print(title, " - Spearman Footrule on Training Set - ", eval_train)
            # print(title, " - Spearman Footrule on Validation Set - ", eval_val)
            # print("Accuracy:  ", trainer.evaluate_classifier(valloader))
            accs.append(trainer.evaluate_classifier(valloader)[0])
            ftrs.append(trainer.evaluate_ranking(valloader))
        print(title)
        return sum(accs) / float(len(accs)), sum(ftrs) / float(len(ftrs))


####################
# PARALLEL METHODS #
####################

def parallel_execution_perform_experiment_accuracy_feynman_pairs(exec_ind: int, data: Dict, counts_input_layer_size: int, onehot_input_layer_size, output_layer_size: int, device) -> Tuple:
    curr_seed = exec_ind + 1
    random.seed(curr_seed)
    np.random.seed(curr_seed)
    torch.manual_seed(curr_seed)
    counts_test_loader = DataLoader(data["counts_test"], batch_size=1, shuffle=True)
    onehot_test_loader = DataLoader(data["onehot_test"], batch_size=1, shuffle=True)
    counts_net = MLPNet(nn.ReLU(), nn.Identity(), counts_input_layer_size, output_layer_size,
                        [220, 140, 80, 26], 0.20)
    onehot_net = MLPNet(nn.ReLU(), nn.Identity(), onehot_input_layer_size, output_layer_size,
                        [220, 140, 80, 26], 0.20)
    counts_trainer = TwoPointsCompareTrainer(counts_net, device, data["counts_training"], False, max_epochs=1)
    onehot_trainer = TwoPointsCompareTrainer(onehot_net, device, data["onehot_training"], False, max_epochs=1)
    counts_trainer.train()
    onehot_trainer.train()
    return NeuralNetEvaluator.evaluate_pairs_classification_accuracy_with_siso_net(counts_trainer.get_net(),
                                                                                   counts_test_loader,
                                                                                   device), NeuralNetEvaluator.evaluate_pairs_classification_accuracy_with_siso_net(
        onehot_trainer.get_net(),
        onehot_test_loader, device)


def parallel_execution_perform_experiment_nn_ranking_online(exec_ind: int, activation_func: Any,
                                                            final_activation_func: Any, input_layer_size: int,
                                                            output_layer_size: int, hidden_layer_sizes: List,
                                                            train_size: int, uncertainty: bool, device: Any,
                                                            X_tr: np.ndarray, y_tr: np.ndarray, verbose: bool,
                                                            valloader: DataLoader, pairs_valloader: DataLoader) -> Tuple:
    curr_seed = exec_ind + 1
    random.seed(curr_seed)
    np.random.seed(curr_seed)
    torch.manual_seed(curr_seed)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                 hidden_layer_sizes, dropout_prob=0.25)
    trainer = OnlineTwoPointsCompareTrainer(net, device, data=None, verbose=False)
    already_seen = []
    for idx in range(train_size):
        if not uncertainty:
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.random_sampler_online(X_tr, y_tr, already_seen)
        else:
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.uncertainty_sampler_online(X_tr, y_tr, trainer,
                                                                                          already_seen)
        pairs_train = NumericalData(pairs_X_tr, pairs_y_tr)
        trainer.change_data(pairs_train)
        loss_epoch_array = trainer.train()
        if verbose and idx == train_size - 1:
            print(f"Loss: {loss_epoch_array[0]}")
    return NeuralNetEvaluator.evaluate_pairs_classification_accuracy_with_siso_net(trainer.get_net(),
                                                                                   pairs_valloader,
                                                                                   device), NeuralNetEvaluator.evaluate_ranking(trainer.get_net(), valloader, device)


def parallel_execution_create_dict_experiment_nn_ranking_online(exec_ind: int, activation_func: Any,
                                                                final_activation_func: Any, input_layer_size: int,
                                                                output_layer_size: int, hidden_layer_sizes: int,
                                                                device: Any, pretrainer_factory: TrainerFactory,
                                                                warmup_data: Dataset, train_size: int,
                                                                uncertainty: bool, X_tr: np.ndarray,
                                                                y_tr: np.ndarray, verbose: bool,
                                                                valloader: DataLoader, repr_plot: str,
                                                                sampl_plot: str, warmup_plot: str) -> List:
    curr_seed = exec_ind + 1
    random.seed(curr_seed)
    np.random.seed(curr_seed)
    torch.manual_seed(curr_seed)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                 hidden_layer_sizes, dropout_prob=0.25)
    trainer = OnlineTwoPointsCompareTrainer(net, device, data=None, verbose=False,
                                            warmup_trainer_factory=pretrainer_factory, warmup_dataset=warmup_data)
    already_seen = []
    results = []
    for idx in range(train_size):
        if not uncertainty:
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.random_sampler_online(X_tr, y_tr, already_seen)
        else:
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.uncertainty_sampler_online(X_tr, y_tr, trainer,
                                                                                          already_seen)
        pairs_train = NumericalData(pairs_X_tr, pairs_y_tr)
        trainer.change_data(pairs_train)
        loss_epoch_array = trainer.train()
        if verbose and idx == train_size - 1:
            print(f"Loss: {loss_epoch_array[0]}")
        this_ftrs = NeuralNetEvaluator.evaluate_ranking(trainer.get_net(), valloader, device)
        results.append((idx + 1, this_ftrs, repr_plot, sampl_plot, warmup_plot))
    return results
