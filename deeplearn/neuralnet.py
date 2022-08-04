from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Any, Callable, Tuple, Dict

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import r2_score, confusion_matrix

from util.Sort import Sort


# ==============================================================================================================
# DATASET AND HANDLERS
# ==============================================================================================================


class TreeData(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, scaler: Any = None):
        self.X = X  # numpy matrix of float values
        self.y = y  # numpy array
        self.scaler = scaler  # e.g., StandardScaler(), already fitted to the training data
        if not(scaler is None):
            self.X = scaler.transform(self.X)  # transform data
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()

    def __len__(self) -> int:
        # gets the number of rows in the dataset
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # gets a data point from the dataset as torch tensor array along with the label
        return self.X[idx], self.y[idx]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        dataset, labels = [], []
        for i in range(len(self)):
            curr_point, curr_label = self[i]
            dataset.append(curr_point.tolist())
            labels.append(curr_label.item())
        return np.array(dataset), np.array(labels)

    def remove_ground_truth_duplicates(self) -> TreeData:
        new_X, new_y = [], []
        for i in range(len(self)):
            is_dupl = False
            j = i+1
            exit_loop = False
            while not(exit_loop) and j < len(self):
                if math.isclose(self[i][1].item(), self[j][1].item(), rel_tol=1e-5):
                    is_dupl = True
                    exit_loop = True
                j += 1
            if not(is_dupl):
                new_X.append(self[i][0].tolist())
                new_y.append(self[i][1].item())
        return TreeData(np.array(new_X, dtype=np.float32), np.array(new_y, dtype=np.float32), scaler=None)


class TreeDataTwoPointsCompare(Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, number_of_points: int, binary_label: bool = False):
        self.dataset = dataset  # this is an instance of a class that inherits torch.utils.data.Dataset class
        self.number_of_points = number_of_points  # how many points we want to generate
        self.original_number_of_points = len(dataset)  # the number of points in dataset instance
        new_data = []
        new_labels = []
        original_data = []
        original_labels = []
        indexes = list(range(self.original_number_of_points))  # [0, 1, 2, ..., self.original_number_of_points-2, self.original_number_of_points-1]
        for _ in range(self.number_of_points):
            exit_loop = False
            while not(exit_loop):
                idx = random.choices(indexes, k=2)  # extract two points at random with replacement (for computational efficiency reasons)
                first_point, first_label = self.dataset[idx[0]]  # first point extracted
                second_point, second_label = self.dataset[idx[1]]  # second point extracted
                if not(math.isclose(first_label.item(), second_label.item(), rel_tol=1e-5)):  # if ground truths are equals, then sample again the two points
                    exit_loop = True
            original_data.append(first_point.tolist())
            original_data.append(second_point.tolist())
            original_labels.append(first_label.item())
            original_labels.append(second_label.item())
            if first_label.item() >= second_label.item():  # first point has a higher score than the second one
                if binary_label:
                    new_labels.append(1)  # close to one when the first point is higher: sigmoid(z_final) >= 0.5
                else:
                    new_labels.append(-1.0)  # if the first point is higher, then the loss decreases: -1*(p1-p2)
            else:  # first point has a lower score than the second one
                if binary_label:
                    new_labels.append(0)  # close to zero when the first point is lower: sigmoid(z_final) < 0.5
                else:
                    new_labels.append(1.0)  # if the second point is higher, then the loss decreases: 1*(p1-p2)
            new_data.append(first_point.tolist() + second_point.tolist())
        self.X = torch.tensor(new_data).float()
        self.y = torch.tensor(new_labels).float()
        self.original_X = np.array(original_data, dtype=np.float32)
        self.original_y = np.array(original_labels, dtype=np.float32)

    def __len__(self) -> int:
        # gets the number of rows in the dataset
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # gets a data point from the dataset as torch tensor array along with the label
        return self.X[idx], self.y[idx]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        dataset, labels = [], []
        for i in range(len(self)):
            curr_point, curr_label = self[i]
            dataset.append(curr_point.tolist())
            labels.append(curr_label.item())
        return np.array(dataset), np.array(labels)

    def to_simple_torch_dataset(self) -> TreeData:
        return TreeData(self.original_X, self.original_y, scaler=None)


# ==============================================================================================================
# MODEL PERFORMANCE
# ==============================================================================================================


def model_accuracy(confusion_matrix: np.ndarray) -> Tuple[float, Dict[int, Dict[str, float]]]:
    N = confusion_matrix.shape[0]
    acc = sum([confusion_matrix[i, i] for i in range(N)])
    class_performance = {i: {} for i in range(N)}
    for i in range(N):
        positive_rate = confusion_matrix[i, i]
        true_positives = confusion_matrix[i, :].sum()
        predicted_positives = confusion_matrix[:, i].sum()
        class_performance[i]["precision"] = positive_rate / predicted_positives if predicted_positives != 0 else 0
        class_performance[i]["recall"] = positive_rate / true_positives if true_positives != 0 else 0
        class_performance[i]["f1"] = (2 * class_performance[i]["precision"] * class_performance[i]["recall"]) / (
                    class_performance[i]["precision"] + class_performance[i]["recall"]) if class_performance[i][
                                                                                               "precision"] + \
                                                                                           class_performance[i][
                                                                                               "recall"] != 0 else 0
    return acc / confusion_matrix.sum(), class_performance


def spearman_footrule(origin: List[Any], estimated: List[Any], equality: Callable[[Any, Any], bool]) -> float:
    distance = 0
    for ie in range(len(origin)):
        exit_loop = False
        it = 0
        while not(exit_loop) and it < len(estimated):
            if equality(origin[ie], estimated[it]):
                distance += abs(ie - it)
                exit_loop = True
            it += 1
    n = len(origin)
    if n % 2 == 0:
        distance *= 3.0/float(np.square(n))
    else:
        distance *= 3.0/float(np.square(n)-1.0)
    return distance


# ==============================================================================================================
# COMPARATOR
# ==============================================================================================================


def random_comparator(point_1, point_2, p):  # here point is the ground truth label and p a probability
    if random.random() < p:
        return point_1 < point_2
    else:
        return not(point_1 < point_2)


def plot_random_ranking(device, dataloader):
    df = {"Probability": [], "Footrule": []}
    for p in np.arange(0, 1.1, 0.1):
        df["Probability"].append(p)
        ll = sum([random_spearman(device, dataloader, p) for _ in range(20)])/20.0
        df["Footrule"].append(ll)
    plot = sns.lineplot(data=df, x="Probability", y="Footrule")
    plt.show()
    return plot


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
    comparator = partial(random_comparator, p=p)
    y_pred, _ = Sort.heapsort(y_true_2, comparator, inplace=False, reverse=False)
    return spearman_footrule(y_true, y_pred, lambda x, y: x == y)


class NeuralNetComparator:
    def __init__(self, net: nn.Module):
        self.__net = net

    def eval(self) -> None:
        self.__net.eval()

    def train(self) -> None:
        self.__net.train()

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return self.__net(data)

    @abstractmethod
    def compare(self, point_1: Any, point_2: Any) -> bool:
        pass


class OneOutputNeuronsComparator(NeuralNetComparator):
    def __init__(self, net: nn.Module):
        super(OneOutputNeuronsComparator, self).__init__(net)

    def compare(self, point_1: Any, point_2: Any) -> bool:
        point_1, point_2 = point_1[0], point_2[0]
        output_1, _ = self.apply(point_1.reshape(1, -1))
        output_1 = output_1[0][0].item()
        output_2, _ = self.apply(point_2.reshape(1, -1))
        output_2 = output_2[0][0].item()
        return output_1 < output_2  # first element is lower than the second one


class TwoOutputNeuronsComparator(NeuralNetComparator):
    def __init__(self, net: nn.Module):
        super(TwoOutputNeuronsComparator, self).__init__(net)

    def compare(self, point_1: Any, point_2: Any) -> bool:
        point_1, point_2 = point_1[0], point_2[0]
        point = torch.cat((point_1, point_2), dim=0).float().reshape(1, -1)
        output, _ = self.apply(point)
        output = output[0]
        output_1, output_2 = output[0].item(), output[1].item()
        return output_1 < output_2  # first element is lower than the second one


class TwoOutputNeuronsSoftmaxComparator(NeuralNetComparator):
    def __init__(self, net: nn.Module):
        super(TwoOutputNeuronsSoftmaxComparator, self).__init__(net)

    def compare(self, point_1: Any, point_2: Any) -> bool:
        point_1, point_2 = point_1[0], point_2[0]
        sm = nn.Softmax(dim=0)
        point = torch.cat((point_1, point_2), dim=0).float().reshape(1, -1)
        output = sm(self.apply(point)[0][0])
        output_1, output_2 = output[0].item(), output[1].item()
        return output_1 >= output_2  # the neural network predicted class 0, it means that first element is lower than the second one


class OneOutputNeuronsSigmoidComparator(NeuralNetComparator):
    def __init__(self, net: nn.Module):
        super(OneOutputNeuronsSigmoidComparator, self).__init__(net)

    def compare(self, point_1: Any, point_2: Any) -> bool:
        point_1, point_2 = point_1[0], point_2[0]
        point = torch.cat((point_1, point_2), dim=0).float().reshape(1, -1)
        output, _ = self.apply(point)
        output = output[0][0].item()
        return output < 0.5  # the neural network predicted class 0, it means that first element is lower than the second one


class NeuralNetComparatorFactory:

    @abstractmethod
    def create(self, net: nn.Module) -> NeuralNetComparator:
        pass


class OneOutputNeuronsComparatorFactory(NeuralNetComparatorFactory):

    def create(self, net: nn.Module) -> NeuralNetComparator:
        return OneOutputNeuronsComparator(net)


class TwoOutputNeuronsComparatorFactory(NeuralNetComparatorFactory):

    def create(self, net: nn.Module) -> NeuralNetComparator:
        return TwoOutputNeuronsComparator(net)


class TwoOutputNeuronsSoftmaxComparatorFactory(NeuralNetComparatorFactory):

    def create(self, net: nn.Module) -> NeuralNetComparator:
        return TwoOutputNeuronsSoftmaxComparator(net)


class OneOutputNeuronsSigmoidComparatorFactory(NeuralNetComparatorFactory):

    def create(self, net: nn.Module) -> NeuralNetComparator:
        return OneOutputNeuronsSigmoidComparator(net)


# ==============================================================================================================
# TRAINER
# ==============================================================================================================


class Trainer(ABC):
    def __init__(self, net, device, data):
        self.device = device
        self.data = data
        self.net = net.to(self.device)

    def model(self):
        return self.net

    @abstractmethod
    def train(self):
        pass

    def evaluate_classifier(self, dataloader):
        total, correct = 0, 0
        y_true = []
        y_pred = []
        self.net.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
                outputs, _ = self.net(inputs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                y_pred.extend(pred.tolist())
                y_true.extend(labels.tolist())
            y_true = list(np.concatenate(y_true).flat)
            y_pred = list(np.concatenate(y_pred).flat)
            cf_matrix = confusion_matrix(y_true, y_pred)
        self.net.train()
        return model_accuracy(cf_matrix)

    def evaluate_regressor(self, dataloader):
        y_true = []
        y_pred = []
        self.net.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
                outputs, _ = self.net(inputs)
                y_pred.extend(outputs.tolist())
                y_true.extend(labels.tolist())
            y_true = list(np.concatenate(y_true).flat)
            y_pred = list(np.concatenate(y_pred).flat)
        self.net.train()
        return r2_score(y_true, y_pred)

    def evaluate_ranking(self, dataloader):
        y_true = []
        y_pred = []
        self.net.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
                outputs, _ = self.net(inputs)
                for i in range(len(inputs)):
                    curr_input, curr_label, curr_output = inputs[i], labels[i][0].item(), outputs[i][0].item()
                    y_true.append((curr_input, curr_label))
                    y_pred.append((curr_input, curr_output))
            y_true, _ = Sort.heapsort(y_true, lambda x, y: x[1] < y[1], inplace=False, reverse=False)
            y_pred, _ = Sort.heapsort(y_pred, lambda x, y: x[1] < y[1], inplace=False, reverse=False)
        self.net.train()
        return spearman_footrule(y_true, y_pred, lambda x, y: torch.equal(x[0], y[0]))

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            X = X.to(self.device)
            res = self.net(X)
        self.net.train()
        return res


class StandardBatchTrainer(Trainer):
    def __init__(self, net, device, dataloader, loss_fn, optimizer_name='adam',
                 is_classification_task=False, verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, max_epochs=20):
        super(StandardBatchTrainer, self).__init__(net, device, dataloader)
        self.is_classification_task = is_classification_task
        self.optimizer_name = optimizer_name
        self.verbose = verbose
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.max_epochs = max_epochs

    def train(self):
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                                  momentum=self.momentum, dampening=self.dampening)
        else:
            raise ValueError(f"{self.optimizer_name} is not a valid value for argument optimizer.")
        loss_epoch_arr = []
        self.net.train()
        for epoch in range(self.max_epochs):
            for batch in self.data:
                inputs, labels = batch
                inputs = inputs.to(self.device).float()
                if self.is_classification_task:
                    labels = labels.to(self.device).long()
                else:
                    labels = labels.to(self.device).float().reshape((labels.shape[0], 1))
                optimizer.zero_grad()
                outputs, _ = self.net(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr


class TwoPointsCompareTrainer(Trainer):
    def __init__(self, net, device, dataloader,
                 verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, max_epochs=20):
        super(TwoPointsCompareTrainer, self).__init__(net, device, dataloader)
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.max_epochs = max_epochs
        self.output_layer_size = net.number_of_output_neurons()
        self.input_layer_size = net.number_of_input_neurons()

    def train(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_epoch_arr = []
        one = torch.tensor(1, dtype=torch.float32)
        minus_one = torch.tensor(-1, dtype=torch.float32)
        for epoch in range(self.max_epochs):
            for batch in self.data:
                self.net.train()
                optimizer.zero_grad()
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
                single_point_dim = inputs.shape[1]//2
                inputs_1 = inputs[:, :single_point_dim]
                inputs_2 = inputs[:, single_point_dim:]
                outputs_1, _ = self.net(inputs_1)
                outputs_1 = outputs_1.flatten()
                loss_1 = one*torch.mean(labels*outputs_1)
                outputs_2, _ = self.net(inputs_2)
                outputs_2 = outputs_2.flatten()
                loss_2 = minus_one*torch.mean(labels*outputs_2)
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr

    def evaluate_classifier(self, dataloader):
        self.net.eval()
        y_true = []
        points = []
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            for i in range(len(inputs)):
                points.append(inputs[i].tolist())
                y_true.append(labels[i][0].item())
        y_true = np.array(y_true, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        ddd = TreeData(points, y_true, scaler=None)
        ddd = TreeDataTwoPointsCompare(ddd, 2000, binary_label=True)
        ddd = DataLoader(ddd, batch_size=1, shuffle=True)
        y_true = []
        y_pred = []
        self.net.eval()
        with torch.no_grad():
            for batch in ddd:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
                single_point_dim = inputs.shape[1] // 2
                inputs_1 = inputs[:, :single_point_dim]
                inputs_2 = inputs[:, single_point_dim:]
                outputs_1, _ = self.net(inputs_1)
                outputs_1 = outputs_1[0][0].item()
                outputs_2, _ = self.net(inputs_2)
                outputs_2 = outputs_2[0][0].item()
                if outputs_1 >= outputs_2:
                    pred = 1
                else:
                    pred = 0
                y_pred.append(pred)
                y_true.append(labels[0].item())
            cf_matrix = confusion_matrix(y_true, y_pred)
        self.net.train()
        return model_accuracy(cf_matrix)


class TwoPointsCompareDoubleInputTrainer(Trainer):
    def __init__(self, net, device, dataloader, comparator_factory, loss_fn, optimizer_name='adam',
                 is_classification_task=False, verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, max_epochs=20):
        super(TwoPointsCompareDoubleInputTrainer, self).__init__(net, device, dataloader)
        self.is_classification_task = is_classification_task
        self.comparator_factory = comparator_factory
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.max_epochs = max_epochs
        self.output_layer_size = net.number_of_output_neurons()
        self.input_layer_size = net.number_of_input_neurons()

    def train(self):
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                                  momentum=self.momentum, dampening=self.dampening)
        else:
            raise ValueError(f"{self.optimizer_name} is not a valid value for argument optimizer.")
        loss_epoch_arr = []
        self.net.train()
        for epoch in range(self.max_epochs):
            for batch in self.data:
                inputs, labels = batch
                inputs = inputs.to(self.device).float()
                if self.is_classification_task:
                    labels = labels.to(self.device).long()
                else:
                    labels = labels.to(self.device).float().reshape((labels.shape[0], 1))
                optimizer.zero_grad()
                outputs, _ = self.net(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr

    def evaluate_classifier(self, dataloader):
        self.net.eval()
        y_true = []
        points = []
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            for i in range(len(inputs)):
                points.append(inputs[i].tolist())
                y_true.append(labels[i][0].item())
        y_true = np.array(y_true, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        ddd = TreeData(points, y_true, scaler=None)
        ddd = TreeDataTwoPointsCompare(ddd, 2000, binary_label=True)
        ddd = DataLoader(ddd, batch_size=1, shuffle=True)
        y_true = []
        y_pred = []
        self.net.eval()
        with torch.no_grad():
            for batch in ddd:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
                outputs, _ = self.net(inputs)
                outputs = outputs[0]
                if len(outputs) == 1:
                    res = outputs[0].item()
                    if res < 0.5:
                        pred = 0
                    else:
                        pred = 1
                elif len(outputs) == 2:
                    res = nn.Softmax(dim=0)(outputs)
                    pred = torch.argmax(res)
                else:
                    raise ValueError(f"{len(outputs)} is different from 1 or 2. The number of output neurons is not 1 or 2.")
                y_pred.append(pred)
                y_true.append(labels[0].item())
            cf_matrix = confusion_matrix(y_true, y_pred)
        self.net.train()
        return model_accuracy(cf_matrix)

    def evaluate_ranking(self, dataloader):
        points = []
        self.net.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
                for i in range(len(inputs)):
                    points.append((inputs[i], labels[i][0].item()))
            y_true, _ = Sort.heapsort(points, lambda x, y: x[1] < y[1], inplace=False, reverse=False)
            comparator = self.comparator_factory.create(self.net)
            y_pred, _ = Sort.heapsort(points, comparator.compare, inplace=False, reverse=False)
        self.net.train()
        return spearman_footrule(y_true, y_pred, lambda x, y: torch.equal(x[0], y[0]))


# ==============================================================================================================
# NEURAL NETWORK MODELS
# ==============================================================================================================


class MLPNet(nn.Module):
    def __init__(self, activation_func: Any, final_activation_func: Any, input_layer_size: int, output_layer_size: int, hidden_layer_sizes: List[int] = [], dropout_prob: float = 0.0):
        super(MLPNet, self).__init__()
        self.activation_func = activation_func  # e.g., nn.ReLU()
        self.final_activation_func = final_activation_func  # e.g., nn.Tanh()
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        fc_components = []
        layer_sizes = hidden_layer_sizes + [output_layer_size]
        curr_dim = input_layer_size
        for i in range(len(layer_sizes)):
            curr_layer = nn.Linear(curr_dim, layer_sizes[i])
            fc_components.append(curr_layer)
            if i != len(layer_sizes) - 1:
                fc_components.append(self.activation_func)
            else:
                fc_components.append(self.final_activation_func)
            if i == len(layer_sizes) - 3 or i == len(layer_sizes) - 5 or i == len(layer_sizes) - 7 or i == len(layer_sizes) - 9:
                fc_components.append(self.dropout)
            curr_dim = layer_sizes[i]

        self.fc_model = nn.Sequential(*fc_components[:-2])
        self.last_layer = nn.Sequential(fc_components[-2], fc_components[-1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        z = x.detach().clone()
        x = self.last_layer(x)

        uncertainty = []
        if not self.training:
            uncert = [self.last_layer(nn.Dropout(self.dropout_prob)(z)) for _ in range(10)]
            for i in range(x.size(0)):
                curr_uncert = []
                for j in range(len(uncert)):
                    curr_uncert.append(uncert[j][i][0].item())
                uncertainty.append(np.std(curr_uncert))
        return x, uncertainty

    def number_of_output_neurons(self) -> int:
        return self.output_layer_size

    def number_of_input_neurons(self) -> int:
        return self.input_layer_size
