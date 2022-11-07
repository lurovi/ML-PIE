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

import torch.multiprocessing as mp
import random

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.dataset.PairSamplerFactory import PairSamplerFactory
from deeplearn.dataset.RandomSamplerWithReplacement import RandomSamplerWithReplacement
from deeplearn.model.DropOutMLPNet import DropOutMLPNet

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

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def perform_experiment_nn_ranking_online(self, encoding_type, ground_truth_type,
                                             amount_of_feedback, activation_func,
                                             final_activation_func, hidden_layer_sizes, device, sampler):
        accs, ftrs = [], []
        verbose = False
        random.seed(self.__starting_seed)
        np.random.seed(self.__starting_seed)
        torch.manual_seed(self.__starting_seed)
        torch.use_deterministic_algorithms(True)

        trees = self.__data_generator.get_X_y(encoding_type, ground_truth_type)
        training, validation, test = trees["training"], trees["validation"], trees["test"]
        input_layer_size = self.__data_generator.get_structure().get_encoder(encoding_type).size()
        output_layer_size = 1
        X_tr, y_tr = training.get_points_and_labels()
        X_va, y_va = validation.get_points_and_labels()
        pairs_X_y_va = RandomSamplerWithReplacement(n_pairs=2000).sample(X_va, y_va)
        valloader = DataLoader(validation, batch_size=1, shuffle=True)
        pairs_valloader = DataLoader(pairs_X_y_va, batch_size=1, shuffle=True)

        pool = mp.Pool(self.__num_repeats if mp.cpu_count() > self.__num_repeats else (mp.cpu_count() - 1))
        exec_func = partial(parallel_execution_perform_experiment_nn_ranking_online, activation_func=activation_func, final_activation_func=final_activation_func, input_layer_size=input_layer_size, output_layer_size=output_layer_size, hidden_layer_sizes=hidden_layer_sizes, amount_of_feedback=amount_of_feedback, sampler=sampler, device=device, X_tr=X_tr, y_tr=y_tr, verbose=verbose, valloader=valloader, pairs_valloader=pairs_valloader)
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
                                             final_activation_func, hidden_layer_sizes, device, sampler,
                                             warmup=None):
        df = {"Amount of feedback": [], "Spearman footrule": [], "Average uncertainty": [],
              "Encoding": [], "Ground-truth": [],
              "Sampling": [], "Warm-up": [], "Seed": []}
        verbose = False
        repr_plot = encoding_type[0].upper() + encoding_type[1:]
        repr_plot = repr_plot.replace("_", " ")
        ground_plot = ground_truth_type[0].upper() + ground_truth_type[1:]
        ground_plot = ground_plot.replace("_", " ")
        sampl_plot = sampler.create_sampler(1, None).get_string_repr()
        sampl_plot = sampl_plot[0].upper() + sampl_plot[1:]
        sampl_plot = sampl_plot.replace("_", " ")

        random.seed(self.__starting_seed)
        np.random.seed(self.__starting_seed)
        torch.manual_seed(self.__starting_seed)
        torch.use_deterministic_algorithms(True)

        trees = self.__data_generator.get_X_y(encoding_type, ground_truth_type)
        training, validation, test = trees["training"], trees["validation"], trees["test"]
        input_layer_size = self.__data_generator.get_structure().get_encoder(encoding_type).size()
        output_layer_size = 1

        X_tr, y_tr = training.get_points_and_labels()

        if warmup is not None:
            warmup_plot = warmup[0].upper() + warmup[1:]
            warmup_plot = warmup_plot.replace("_", " ")
            warmup_data = self.__data_generator.get_warm_up_data(encoding_type, warmup)
            pretrainer_factory = TwoPointsCompareTrainerFactory(False, 1)
        else:
            warmup_plot = "No Warm-up"
            warmup_data = None
            pretrainer_factory = None

        pool = mp.Pool(self.__num_repeats if mp.cpu_count() > self.__num_repeats else (mp.cpu_count() - 1), maxtasksperchild=1)
        exec_func = partial(parallel_execution_create_dict_experiment_nn_ranking_online,
                            activation_func=activation_func, final_activation_func=final_activation_func, input_layer_size=input_layer_size,
                                                                output_layer_size=output_layer_size, hidden_layer_sizes=hidden_layer_sizes,
                                                                device=device, pretrainer_factory=pretrainer_factory,
                                                                warmup_data=warmup_data, amount_of_feedback=amount_of_feedback,
                                                                sampler=sampler, X_tr=X_tr,
                                                                y_tr=y_tr, verbose=verbose,
                                                                valloader=validation, repr_plot=repr_plot, ground_plot=ground_plot,
                                                                sampl_plot=sampl_plot, warmup_plot=warmup_plot
                                                                )
        exec_res = pool.map(exec_func, list(range(self.__starting_seed, self.__starting_seed + self.__num_repeats)))
        pool.close()
        pool.join()
        for l in exec_res:
            for curr_train_index, curr_ftrs, curr_uncert, curr_repr_plot, curr_ground_plot, curr_sampl_plot, curr_warmup_plot, curr_seed_val in l:
                df["Amount of feedback"].append(curr_train_index)
                df["Spearman footrule"].append(curr_ftrs)
                df["Average uncertainty"].append(curr_uncert)
                df["Encoding"].append(curr_repr_plot)
                df["Ground-truth"].append(curr_ground_plot)
                df["Sampling"].append(curr_sampl_plot)
                df["Warm-up"].append(curr_warmup_plot)
                df["Seed"].append(curr_seed_val)
        print("Executed: "+repr_plot+" - "+ground_plot+" - "+sampl_plot+" - "+warmup_plot+".")
        return df

    def perform_experiment_accuracy_feynman_pairs(self, folder, device):

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

        pool = mp.Pool(self.__num_repeats if mp.cpu_count() > self.__num_repeats else (mp.cpu_count() - 1))
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
                                                            amount_of_feedback: int, sampler: PairSamplerFactory, device: Any,
                                                            X_tr: torch.Tensor, y_tr: torch.Tensor, verbose: bool,
                                                            valloader: DataLoader, pairs_valloader: DataLoader) -> Tuple:
    curr_seed = exec_ind
    random.seed(curr_seed)
    np.random.seed(curr_seed)
    torch.manual_seed(curr_seed)

    net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                 hidden_layer_sizes, dropout_prob=0.25)
    trainer = OnlineTwoPointsCompareTrainer(net, device, data=None, verbose=False)
    samplerrr = sampler.create_sampler(1)
    for idx in range(amount_of_feedback):
        pairs_train = samplerrr.sample(X_tr, y_tr, trainer)
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
                                                                sampler: PairSamplerFactory, X_tr: torch.Tensor,
                                                                y_tr: torch.Tensor, verbose: bool,
                                                                valloader: Dataset, repr_plot: str, ground_plot: str,
                                                                sampl_plot: str, warmup_plot: str
                                                                ) -> List:
    curr_seed = exec_ind
    random.seed(curr_seed)
    np.random.seed(curr_seed)
    torch.manual_seed(curr_seed)

    g_valloader = torch.Generator()
    g_valloader.manual_seed(curr_seed)
    valloader = DataLoader(valloader, batch_size=1, shuffle=True, num_workers=0, worker_init_fn=ExpsExecutor.seed_worker, generator=g_valloader)

    # net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size, hidden_layer_sizes)
    net = DropOutMLPNet(activation_func, final_activation_func, input_layer_size)
    trainer = OnlineTwoPointsCompareTrainer(net, device, data=None, verbose=False,
                                            warmup_trainer_factory=pretrainer_factory, warmup_dataset=warmup_data)

    results = []
    samplerrr = sampler.create_sampler(1)
    for idx in range(amount_of_feedback):
        pairs_train = samplerrr.sample(X_tr, y_tr, trainer)
        trainer.change_data(pairs_train)
        loss_epoch_array = trainer.fit()
        if verbose and idx == amount_of_feedback - 1:
            print(f"Loss: {loss_epoch_array[0]}")
        this_ftrs = NeuralNetEvaluator.evaluate_ranking(trainer.get_net(), valloader, device)
        this_uncert = NeuralNetEvaluator.evaluate_average_uncertainty(trainer.get_net(), valloader, device)
        results.append((idx + 1, this_ftrs, this_uncert, repr_plot, ground_plot, sampl_plot, warmup_plot, curr_seed))
    return results
