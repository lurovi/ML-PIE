import random
from abc import ABC, abstractmethod
from functools import partial
from typing import List

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
from util.sort import heapsort


# ==============================================================================================================
# DATASET AND HANDLERS
# ==============================================================================================================


class TreeData(Dataset):
    def __init__(self, X, y, scaler=None):
        self.X = X  # numpy matrix of float values
        self.y = y  # numpy array
        self.scaler = scaler  # e.g., StandardScaler(), already fitted to the training data
        if not(scaler is None):
            self.X = scaler.transform(self.X)  # transform data
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()

    def __len__(self):
        # gets the number of rows in the dataset
        return len(self.y)

    def __getitem__(self, idx):
        # gets a data point from the dataset as torch tensor array along with the label
        return self.X[idx], self.y[idx]

    def to_numpy(self):
        dataset, labels = [], []
        for i in range(len(self)):
            curr_point, curr_label = self[i]
            dataset.append(curr_point.tolist())
            labels.append(curr_label.item())
        return np.array(dataset), np.array(labels)


class TreeDataTwoPointsCompare(Dataset):
    def __init__(self, dataset, number_of_points, binary_label=False):
        self.dataset = dataset  # this is an instance of a class that inherits torch.utils.data.Dataset class
        self.number_of_points = number_of_points  # how many points we want to generate
        self.original_number_of_points = len(dataset)  # the number of points in dataset instance
        new_data = []
        new_labels = []
        indexes = list(range(self.original_number_of_points))  # [0, 1, 2, ..., self.original_number_of_points-2, self.original_number_of_points-1]
        for _ in range(self.number_of_points):
            idx = random.choices(indexes, k=2)  # extract two points at random with replacement (for computational efficiency reasons)
            first_point, first_label = self.dataset[idx[0]]  # first point extracted
            second_point, second_label = self.dataset[idx[1]]  # second point extracted
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

    def __len__(self):
        # gets the number of rows in the dataset
        return len(self.y)

    def __getitem__(self, idx):
        # gets a data point from the dataset as torch tensor array along with the label
        return self.X[idx], self.y[idx]

    def to_numpy(self):
        dataset, labels = [], []
        for i in range(len(self)):
            curr_point, curr_label = self[i]
            dataset.append(curr_point.tolist())
            labels.append(curr_label.item())
        return np.array(dataset), np.array(labels)


# ==============================================================================================================
# MODEL PERFORMANCE
# ==============================================================================================================


def spearman_footrule(y_true, y_pred):
    # y_true (or y_pred) is a list of scalar values
    # each argument is arg sorted in reverse order, i.e., the first element is the index of the greatest value in each list
    y_true = np.argsort(y_true, kind="heapsort")[::-1]
    y_pred = np.argsort(y_pred, kind="heapsort")[::-1]
    return spearman_footrule_direct(y_true, y_pred)  # apply spearman footrule directly


def spearman_footrule_direct(y_true, y_pred):
    # y_true (or y_pred) is a list of integer indexes which represents a sorting of a list of scalar values
    r = len(y_true)  # number of values
    # compute normalization factor
    if r % 2 == 0:  # even number of values
        r = r ** 2
    else:  # odd number of values
        r = r ** 2 - 1
    return (3.0 / float(r)) * np.absolute(np.subtract(y_true, y_pred)).sum()  # factor * sum( |y_true_argsorted - y_pred_argsorted| )


# ==============================================================================================================
# LOSS
# ==============================================================================================================


def two_points_compare_loss(outputs, labels):
    return torch.sum( labels[:, 0] * (outputs[:, 0] - outputs[:, 1]) )  # sum( labels*(first_prediction - second_prediction) ) where labels can be either -1 or 1.


# ==============================================================================================================
# COMPARATOR
# ==============================================================================================================


def neuralnet_one_output_neurons_comparator(point_1, point_2, neural_network):
    neural_network.eval()
    output_1 = neural_network(point_1.reshape(1, -1))[0][0]
    output_2 = neural_network(point_2.reshape(1, -1))[0][0]
    return output_1.item() < output_2.item()  # first element is lower than the second one


def neuralnet_two_output_neurons_comparator(point_1, point_2, neural_network):
    neural_network.eval()
    point = torch.cat((point_1, point_2), dim=0).float().reshape(1, -1)
    output = neural_network(point)[0]
    return output[0].item() < output[1].item()  # first element is lower than the second one


def neuralnet_two_output_neurons_softmax_comparator(point_1, point_2, neural_network):
    neural_network.eval()
    sm = nn.Softmax(dim=0)
    point = torch.cat((point_1, point_2), dim=0).float().reshape(1, -1)
    output = sm(neural_network(point)[0])
    return output[0].item() >= output[1].item()  # the neural network predicted class 0, it means that first element is lower than the second one


def neuralnet_one_output_neurons_sigmoid_comparator(point_1, point_2, neural_network):
    neural_network.eval()
    point = torch.cat((point_1, point_2), dim=0).float().reshape(1, -1)
    output = neural_network(point)[0]
    return output[0].item() < 0.5  # the neural network predicted class 0, it means that first element is lower than the second one


def neuralnet_one_output_neurons_tanh_comparator(point_1, point_2, neural_network):
    neural_network.eval()
    point = torch.cat((point_1, point_2), dim=0).float().reshape(1, -1)
    output = neural_network(point)[0]
    return output[0].item() < 0.0  # the neural network predicted class 0, it means that first element is lower than the second one


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
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            outputs = self.net(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            y_pred.extend(pred.tolist())
            y_true.extend(labels.tolist())
        y_true = list(np.concatenate(y_true).flat)
        y_pred = list(np.concatenate(y_pred).flat)
        cf_matrix = confusion_matrix(y_true, y_pred)
        return model_accuracy(cf_matrix)

    def evaluate_regressor(self, dataloader):
        y_true = []
        y_pred = []
        self.net.eval()
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            outputs = self.net(inputs)
            y_pred.extend(outputs.tolist())
            y_true.extend(labels.tolist())
        y_true = list(np.concatenate(y_true).flat)
        y_pred = list(np.concatenate(y_pred).flat)
        return r2_score(y_true, y_pred)

    def evaluate_ranking(self, dataloader):
        y_true = []
        y_pred = []
        self.net.eval()
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            outputs = self.net(inputs)
            y_pred.extend(outputs.tolist())
            y_true.extend(labels.tolist())
        y_true = list(np.concatenate(y_true).flat)
        y_pred = list(np.concatenate(y_pred).flat)
        return spearman_footrule(y_true, y_pred)

    def predict(self, X):
        self.net.eval()
        X = X.to(self.device)
        return self.net(X)


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
                outputs = self.net(inputs)
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
                outputs_1 = self.net(inputs_1).flatten()
                loss_1 = one*torch.mean(labels*outputs_1)
                outputs_2 = self.net(inputs_2).flatten()
                loss_2 = minus_one*torch.mean(labels*outputs_2)
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr

    def evaluate_ranking(self, dataloader):
        y_true = []
        points = []
        self.net.eval()
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            for i in range(len(inputs)):
                points.append(inputs[i])
                y_true.append(labels[i][0].item())
        y_true = np.argsort(y_true, kind="heapsort")[::-1]
        comparator = partial(neuralnet_one_output_neurons_comparator, neural_network=self.net)
        _, y_pred = heapsort(points, comparator, inplace=False, reverse=True)
        return spearman_footrule_direct(y_true, np.array(y_pred))


class TwoPointsCompareDoubleInputTrainer(Trainer):
    def __init__(self, net, device, dataloader, comparator_fn, loss_fn, optimizer_name='adam',
                 is_classification_task=False, verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, max_epochs=20):
        super(TwoPointsCompareDoubleInputTrainer, self).__init__(net, device, dataloader)
        self.is_classification_task = is_classification_task
        self.comparator_fn = comparator_fn
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.verbose = verbose
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
                outputs = self.net(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr

    def evaluate_ranking(self, dataloader):
        y_true = []
        points = []
        self.net.eval()
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            for i in range(len(inputs)):
                points.append(inputs[i])
                y_true.append(labels[i][0].item())
        y_true = np.argsort(y_true, kind="heapsort")[::-1]
        comparator = partial(self.comparator_fn, neural_network=self.net)
        _, y_pred = heapsort(points, comparator, inplace=False, reverse=True)
        return spearman_footrule_direct(y_true, np.array(y_pred))


# ==============================================================================================================
# NEURAL NETWORK MODELS
# ==============================================================================================================


class LeNet(nn.Module):
    def __init__(self, activation_func=nn.ReLU(), in_channels=1, fc_input_layer_size=256):
        super(LeNet, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.activation_func = activation_func
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(6, 16, kernel_size=5)
        self.linear_1 = nn.Linear(fc_input_layer_size, 120)
        self.linear_2 = nn.Linear(120, 84)
        self.linear_3 = nn.Linear(84, 10)
        self.cnn_model = nn.Sequential(
             self.conv2d_1,  # (N, 1, 28, 28) -> (N, 6, 24, 24)
             self.activation_func,
             self.maxpool2d,  # (N, 6, 24, 24) -> (N, 6, 12, 12)
             self.conv2d_2,  # (N, 6, 12, 12) -> (N, 6, 8, 8)
             self.activation_func,
             self.maxpool2d)  # (N, 6, 8, 8) -> (N, 16, 4, 4)
        self.fc_model = nn.Sequential(
             self.linear_1,  # (N, 256) -> (N, 120)
             self.activation_func,
             self.linear_2,  # (N, 120) -> (N, 84)
             self.activation_func,
             self.linear_3)  # (N, 84)  -> (N, 10)) #10 classes

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        emb = x.tolist()
        x = self.fc_model(x)
        uncertainity = ( ( 1.0 - torch.std( nn.Softmax(dim=1)(x), dim=1, unbiased=False )  )*100 ).tolist()
        return x, emb, uncertainity


class MLPNet(nn.Module):
    def __init__(self, activation_func, final_activation_func, input_layer_size: int, output_layer_size: int, hidden_layer_sizes: List[int] = [], dropout_prob: float = 0.0):
        super(MLPNet, self).__init__()
        self.activation_func = activation_func  # e.g., nn.ReLU()
        self.final_activation_func = final_activation_func  # e.g., nn.Tanh()
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.dropout_prob = dropout_prob

        linear_layers = []
        layer_sizes = hidden_layer_sizes + [output_layer_size]
        curr_dim = input_layer_size
        for i in range(len(layer_sizes)):
            linear_layers.append(nn.Linear(curr_dim, layer_sizes[i]))
            if i != len(layer_sizes) - 1:
                linear_layers.append(self.activation_func)
            else:
                linear_layers.append(self.final_activation_func)
            if i == len(layer_sizes) - 3 or i == len(layer_sizes) - 5 or i == len(layer_sizes) - 7 or i == len(layer_sizes) - 9:
                linear_layers.append(nn.Dropout(self.dropout_prob))
            curr_dim = layer_sizes[i]

        self.fc_model = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


class SymmetricAutoEncoder(nn.Module):
    def __init__(self, activation_func, final_activation_func, input_layer_size: int, bottleneck_size: int, hidden_layer_sizes: List[int] = [], dropout_probability: float = 0.0):
        super(SymmetricAutoEncoder, self).__init__()
        self.activation_func = activation_func  # e.g., nn.ReLU()
        self.final_activation_func = final_activation_func  # e.g., nn.Tanh()
        self.input_layer_size = input_layer_size
        self.bottleneck_size = bottleneck_size
        self.dropout_probability = dropout_probability

        linear_layers = []
        layer_sizes = hidden_layer_sizes + [bottleneck_size]
        curr_dim = input_layer_size
        for i in range(len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                linear_layers.append(nn.Dropout(self.dropout_probability))
            linear_layers.append(nn.Linear(curr_dim, layer_sizes[i]))
            linear_layers.append(self.activation_func)
            curr_dim = layer_sizes[i]
        layer_sizes = reversed([input_layer_size] + hidden_layer_sizes)
        curr_dim = bottleneck_size
        for i in range(len(layer_sizes)):
            linear_layers.append(nn.Linear(curr_dim, layer_sizes[i]))
            if i == 0:
                linear_layers.append(self.activation_func)
                linear_layers.append(nn.Dropout(self.dropout_probability))
            elif i == len(layer_sizes) - 1:
                linear_layers.append(self.final_activation_func)
            else:
                linear_layers.append(self.activation_func)
            curr_dim = layer_sizes[i]

        self.fc_model = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


class FourLayerNet(nn.Module):
    def __init__(self, activation_func, final_activation_func, fc_input_layer_size=69):
        super(FourLayerNet, self).__init__()
        self.activation_func = activation_func  # e.g., nn.ReLU()
        self.final_activation_func = final_activation_func  # e.g., nn.Tanh()
        self.linear_1 = nn.Linear(fc_input_layer_size, 300)
        self.linear_2 = nn.Linear(300, 110)
        self.linear_3 = nn.Linear(110, 40)
        self.linear_4 = nn.Linear(40, 1)
        self.fc_model = nn.Sequential(
             self.linear_1,
             self.activation_func,
             self.linear_2,
             self.activation_func,
             self.linear_3,
             self.activation_func,
             self.linear_4,
             self.final_activation_func
             )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


class FiveLayerNet(nn.Module):
    def __init__(self, activation_func, final_activation_func, fc_input_layer_size=255):
        super(FiveLayerNet, self).__init__()
        self.activation_func = activation_func  # e.g., nn.ReLU()
        self.final_activation_func = final_activation_func  # e.g., nn.Tanh()
        self.linear_1 = nn.Linear(fc_input_layer_size, 400)
        self.linear_2 = nn.Linear(400, 300)
        self.linear_3 = nn.Linear(300, 200)
        self.linear_4 = nn.Linear(200, 100)
        self.linear_5 = nn.Linear(100, 1)
        self.fc_model = nn.Sequential(
             self.linear_1,
             self.activation_func,
             self.linear_2,
             self.activation_func,
             self.linear_3,
             self.activation_func,
             self.linear_4,
             self.activation_func,
             self.linear_5,
             self.final_activation_func
             )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


class SevenLayerNet(nn.Module):
    def __init__(self, activation_func, final_activation_func, fc_input_layer_size=5355):
        super(SevenLayerNet, self).__init__()
        self.activation_func = activation_func  # e.g., nn.ReLU()
        self.final_activation_func = final_activation_func  # e.g., nn.Tanh()
        self.linear_1 = nn.Linear(fc_input_layer_size, 7000)
        self.linear_2 = nn.Linear(7000, 5000)
        self.linear_3 = nn.Linear(5000, 3500)
        self.linear_4 = nn.Linear(3500, 1500)
        self.linear_5 = nn.Linear(1500, 500)
        self.linear_6 = nn.Linear(500, 100)
        self.linear_7 = nn.Linear(100, 1)
        self.fc_model = nn.Sequential(
            self.linear_1,
            self.activation_func,
            self.linear_2,
            self.activation_func,
            self.linear_3,
            self.activation_func,
            self.linear_4,
            self.activation_func,
            self.linear_5,
            self.activation_func,
            self.linear_6,
            self.activation_func,
            self.linear_7,
            self.final_activation_func
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x
