import os
from functools import partial
from typing import Dict, List, Any, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import multiprocessing
import random

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler

from deeplearn.model.MLPNet import MLPNet
from deeplearn.model.NeuralNetEvaluator import NeuralNetEvaluator

from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer

from deeplearn.trainer.TrainerFactory import TrainerFactory

from deeplearn.trainer.TwoPointsCompareTrainer import TwoPointsCompareTrainer
from deeplearn.trainer.TwoPointsCompareTrainerFactory import TwoPointsCompareTrainerFactory
from exps.DatasetGenerator import DatasetGenerator
from util.EvaluationMetrics import EvaluationMetrics
from util.PicklePersist import PicklePersist
from util.Sort import Sort


class ExpsExecutor:
    def __init__(self, data_generator: DatasetGenerator,
                 starting_seed: int,
                 num_repeats: int):
        self.__data_generator = data_generator
        self.__starting_seed = starting_seed
        self.__num_repeats = num_repeats

    def perform_experiment_nn_ranking_online(self, encoding_type, ground_truth_type,
                                             amount_of_feedback, activation_func,
                                             final_activation_func, hidden_layer_sizes, device, uncertainty=False):
        accs, ftrs = [], []
        verbose = False
        random.seed(self.__starting_seed)
        np.random.seed(self.__starting_seed)
        torch.manual_seed(self.__starting_seed)
        torch.use_deterministic_algorithms(True)

        trees = self.__data_generator.get_X_y(encoding_type, ground_truth_type)
        training, validation, test = trees["training"], trees["validation"], trees["test"]
        input_layer_size = self.__data_generator.get_structure().get_encoding_size(encoding_type)
        output_layer_size = 1
        X_tr, y_tr = training.get_points_and_labels()
        X_va, y_va = validation.get_points_and_labels()
        pairs_X_va, pairs_y_va = PairSampler.random_sampler_with_replacement(X_va, y_va, 2000)
        valloader = DataLoader(validation, batch_size=1, shuffle=True)
        pairs_valloader = DataLoader(NumericalData(pairs_X_va, pairs_y_va), batch_size=1, shuffle=True)

        pool = multiprocessing.Pool(os.cpu_count() - 1)
        exec_func = partial(parallel_execution_perform_experiment_nn_ranking_online, activation_func=activation_func, final_activation_func=final_activation_func, input_layer_size=input_layer_size, output_layer_size=output_layer_size, hidden_layer_sizes=hidden_layer_sizes, amount_of_feedback=amount_of_feedback, uncertainty=uncertainty, device=device, X_tr=X_tr, y_tr=y_tr, verbose=verbose, valloader=valloader, pairs_valloader=pairs_valloader)
        exec_res = pool.map(exec_func, list(range(self.__starting_seed, self.__starting_seed + self.__num_repeats)))
        pool.close()
        pool.join()
        for curr_acc, curr_ftrs in exec_res:
            accs.append(curr_acc)
            ftrs.append(curr_ftrs)
        print(encoding_type + " " + ground_truth_type)

        return sum(accs) / float(len(accs)), sum(ftrs) / float(len(ftrs))

    @staticmethod
    def perform_execution(device):
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

    def create_dict_experiment_nn_ranking_online(self, folder, encoding_type, ground_truth_type,
                                                 amount_of_feedback, activation_func,
                                             final_activation_func, hidden_layer_sizes, device, sampling="random",
                                             warmup=None):
        df = {"Amount of feedback": [], "Spearman footrule": [], "Encoding": [], "Ground-truth": [],
              "Sampling": [], "Warm-up": []}
        verbose = False
        repr_plot = encoding_type[0].upper() + encoding_type[1:]
        repr_plot = repr_plot.replace("_", " ")
        ground_plot = ground_truth_type[0].upper() + ground_truth_type[1:]
        ground_plot = ground_plot.replace("_", " ")
        sampl_plot = sampling[0].upper() + sampling[1:]
        sampl_plot = sampl_plot.replace("_", " ")

        random.seed(self.__starting_seed)
        np.random.seed(self.__starting_seed)
        torch.manual_seed(self.__starting_seed)
        torch.use_deterministic_algorithms(True)

        trees = self.__data_generator.get_X_y(encoding_type, ground_truth_type)
        training, validation, test = trees["training"], trees["validation"], trees["test"]
        input_layer_size = self.__data_generator.get_structure().get_encoding_size(encoding_type)
        output_layer_size = 1

        X_tr, y_tr = training.get_points_and_labels()

        valloader = DataLoader(validation, batch_size=1, shuffle=True)

        if warmup is not None:
            if warmup == "feynman":
                warmup_plot = "Feynman warm-up"
                warmup_data = PicklePersist.decompress_pickle(folder + "/feynman_pairs.pbz2")[encoding_type]["training"]
            elif warmup == "elastic":
                warmup_plot = "Elastic warm-up"
                warmup_data = self.__data_generator.get_warm_up_data(encoding_type, "elastic_model")
            else:
                raise AttributeError("Bad warmup parameter.")
            pretrainer_factory = TwoPointsCompareTrainerFactory(False, 1)
        else:
            warmup_plot = "No Warm-up"
            warmup_data = None
            pretrainer_factory = None

        pool = multiprocessing.Pool(os.cpu_count() - 1)
        exec_func = partial(parallel_execution_create_dict_experiment_nn_ranking_online,
                            activation_func=activation_func, final_activation_func=final_activation_func, input_layer_size=input_layer_size,
                                                                output_layer_size=output_layer_size, hidden_layer_sizes=hidden_layer_sizes,
                                                                device=device, pretrainer_factory=pretrainer_factory,
                                                                warmup_data=warmup_data, amount_of_feedback=amount_of_feedback,
                                                                sampling=sampling, X_tr=X_tr,
                                                                y_tr=y_tr, verbose=verbose,
                                                                valloader=valloader, repr_plot=repr_plot, ground_plot=ground_plot,
                                                                sampl_plot=sampl_plot, warmup_plot=warmup_plot)
        exec_res = pool.map(exec_func, list(range(self.__starting_seed, self.__starting_seed + self.__num_repeats)))
        pool.close()
        pool.join()
        for l in exec_res:
            for curr_train_index, curr_ftrs, curr_repr_plot, curr_ground_plot, curr_sampl_plot, curr_warmup_plot in l:
                df["Amount of feedback"].append(curr_train_index)
                df["Spearman footrule"].append(curr_ftrs)
                df["Encoding"].append(curr_repr_plot)
                df["Ground-truth"].append(curr_ground_plot)
                df["Sampling"].append(curr_sampl_plot)
                df["Warm-up"].append(curr_warmup_plot)
        print("Executed: "+repr_plot+" - "+ground_plot+" - "+sampl_plot+" - "+warmup_plot+".")
        return df

    @staticmethod
    def perform_experiment_accuracy_feynman_pairs(folder, device):

        counts_accs, counts_ftrs, onehot_accs, onehot_ftrs = [], [], [], []
        verbose = False

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.use_deterministic_algorithms(True)

        data = PicklePersist.decompress_pickle(folder + "/feynman_pairs.pbz2")
        counts_input_layer_size = len(data["counts_training"][0][0]) // 2
        onehot_input_layer_size = len(data["onehot_training"][0][0]) // 2
        print(counts_input_layer_size)
        print(onehot_input_layer_size)
        output_layer_size = 1
        print(data["counts_training"].count_labels())
        print(data["counts_test"].count_labels())

        pool = multiprocessing.Pool(4)
        exec_func = partial(parallel_execution_perform_experiment_accuracy_feynman_pairs,
                            data=data, counts_input_layer_size=counts_input_layer_size,
                            onehot_input_layer_size=onehot_input_layer_size, output_layer_size=output_layer_size,
                            device=device)

        exec_res = pool.map(exec_func, list(range(10)))
        pool.close()
        pool.join()
        for curr_acc_counts, curr_acc_onehot in exec_res:
            counts_accs.append(curr_acc_counts)
            onehot_accs.append(curr_acc_onehot)

        return sum(counts_accs) / float(len(counts_accs)), sum(onehot_accs) / float(len(onehot_accs))

####################
# PARALLEL METHODS #
####################


def parallel_execution_perform_experiment_accuracy_feynman_pairs(exec_ind: int, data: Dict, counts_input_layer_size: int, onehot_input_layer_size, output_layer_size: int, device) -> Tuple:
    curr_seed = exec_ind
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
    counts_trainer.fit()
    onehot_trainer.fit()
    return NeuralNetEvaluator.evaluate_pairs_classification_accuracy_with_siso_net(counts_trainer.get_net(),
                                                                                   counts_test_loader,
                                                                                   device), NeuralNetEvaluator.evaluate_pairs_classification_accuracy_with_siso_net(
        onehot_trainer.get_net(),
        onehot_test_loader, device)


def parallel_execution_perform_experiment_nn_ranking_online(exec_ind: int, activation_func: Any,
                                                            final_activation_func: Any, input_layer_size: int,
                                                            output_layer_size: int, hidden_layer_sizes: List,
                                                            amount_of_feedback: int, uncertainty: bool, device: Any,
                                                            X_tr: np.ndarray, y_tr: np.ndarray, verbose: bool,
                                                            valloader: DataLoader, pairs_valloader: DataLoader) -> Tuple:
    curr_seed = exec_ind
    random.seed(curr_seed)
    np.random.seed(curr_seed)
    torch.manual_seed(curr_seed)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                 hidden_layer_sizes, dropout_prob=0.25)
    trainer = OnlineTwoPointsCompareTrainer(net, device, data=None, verbose=False)
    already_seen = []
    for idx in range(amount_of_feedback):
        if not uncertainty:
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.random_sampler_online(X_tr, y_tr, already_seen)
        else:
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.uncertainty_sampler_online(X_tr, y_tr, trainer,
                                                                                          already_seen)
        pairs_train = NumericalData(pairs_X_tr, pairs_y_tr)
        trainer.change_data(pairs_train)
        loss_epoch_array = trainer.fit()
        if verbose and idx == amount_of_feedback - 1:
            print(f"Loss: {loss_epoch_array[0]}")
    return NeuralNetEvaluator.evaluate_pairs_classification_accuracy_with_siso_net(trainer.get_net(),
                                                                                   pairs_valloader,
                                                                                   device), NeuralNetEvaluator.evaluate_ranking(trainer.get_net(), valloader, device)


def parallel_execution_create_dict_experiment_nn_ranking_online(exec_ind: int, activation_func: Any,
                                                                final_activation_func: Any, input_layer_size: int,
                                                                output_layer_size: int, hidden_layer_sizes: int,
                                                                device: Any, pretrainer_factory: TrainerFactory,
                                                                warmup_data: Dataset, amount_of_feedback: int,
                                                                sampling: str, X_tr: np.ndarray,
                                                                y_tr: np.ndarray, verbose: bool,
                                                                valloader: DataLoader, repr_plot: str, ground_plot: str,
                                                                sampl_plot: str, warmup_plot: str) -> List:
    curr_seed = exec_ind
    random.seed(curr_seed)
    np.random.seed(curr_seed)
    torch.manual_seed(curr_seed)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                 hidden_layer_sizes, dropout_prob=0.25)
    trainer = OnlineTwoPointsCompareTrainer(net, device, data=None, verbose=False,
                                            warmup_trainer_factory=pretrainer_factory, warmup_dataset=warmup_data)
    already_seen = []
    results = []
    for idx in range(amount_of_feedback):
        if sampling == "random":
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.random_sampler_online(X_tr, y_tr, already_seen)
        elif sampling == "uncertainty":
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.uncertainty_sampler_online(X_tr, y_tr, trainer,
                                                                                          already_seen)
        elif sampling == "uncertainty_L2":
            pairs_X_tr, pairs_y_tr, already_seen = PairSampler.uncertainty_L2_sampler_online(X_tr, y_tr, trainer,
                                                                                          already_seen)
        else:
            raise AttributeError(f"{sampling} is not a valid sampling criterion.")
        pairs_train = NumericalData(pairs_X_tr, pairs_y_tr)
        trainer.change_data(pairs_train)
        loss_epoch_array = trainer.fit()
        if verbose and idx == amount_of_feedback - 1:
            print(f"Loss: {loss_epoch_array[0]}")
        this_ftrs = NeuralNetEvaluator.evaluate_ranking(trainer.get_net(), valloader, device)
        results.append((idx + 1, this_ftrs, repr_plot, ground_plot, sampl_plot, warmup_plot))
    return results
