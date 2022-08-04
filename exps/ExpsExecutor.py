from functools import partial

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from deeplearn.model.MLPNet import MLPNet
from deeplearn.trainer.TwoPointsCompareTrainer import TwoPointsCompareTrainer
from util.EvaluationMetrics import EvaluationMetrics
from util.PicklePersist import PicklePersist
from util.Sort import Sort
from util.TorchSeedWorker import TorchSeedWorker


class ExpsExecutor:

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

    @staticmethod
    def execute_experiment_nn_ranking(title, generator_data_loader, file_name_training, file_name_dataset, train_size, activation_func,
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
    def example_execution_1(generator_data_loader, device):
        activations = {"identity": nn.Identity(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        for target in ["weights_sum"]:
            for i in range(1, 4+1):
                i_str = str(i)
                for representation in ["counts", "onehot"]:
                    for final_activation in ["identity", "sigmoid", "tanh"]:
                        print(ExpsExecutor.execute_experiment_nn_ranking(
                            target + " " + i_str + " " + representation + " " + final_activation,
                            generator_data_loader,
                            "data/" + representation + "_" + target + "_" + "trees_twopointscompare" + "_" + i_str + ".pbz2",
                            "data/" + representation + "_" + target + "_" + "trees" + "_" + i_str + ".pbz2", 200,
                            nn.ReLU(), activations[final_activation],
                            [220, 140, 80, 26], device, max_epochs=1, batch_size=1
                        ))

    @staticmethod
    def example_execution_2(generator_data_loader, device):
        activations = {"identity": nn.Identity(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        for target in ["weights_sum"]:
            for i in range(1, 10+1):
                i_str = str(i)
                for representation in ["counts", "onehot"]:
                    for final_activation in ["identity"]:
                        for criterion in ["target", "nodes"]:
                            twopointscomparename = "trees_twopointscompare" if criterion == "target" else "trees_twopointscompare_samenodes"
                            print(ExpsExecutor.execute_experiment_nn_ranking(
                                target + " " + i_str + " " + representation + " " + final_activation,
                                generator_data_loader,
                                "data/" + representation + "_" + target + "_" + twopointscomparename + "_" + i_str + ".pbz2",
                                "data/" + representation + "_" + target + "_" + "trees" + "_" + i_str + ".pbz2", 200,
                                nn.ReLU(), activations[final_activation],
                                [220, 140, 80, 26], device, max_epochs=1, batch_size=1
                            ))
