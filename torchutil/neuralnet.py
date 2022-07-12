import random
from abc import ABC, abstractmethod
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


# ==============================================================================================================
# DATASET AND HANDLERS
# ==============================================================================================================


class TreeData(Dataset):
    def __init__(self, X, y, scaler=None):
        self.X = X  # numpy matrix of float values
        self.y = y  # numpy array
        self.scaler = scaler  # e.g., StandardScaler(), already fitted to the training data
        if not(scaler is None):
            self.X = scaler.transform(self.X)
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()

    def __len__(self):
        # gets the number of rows in the dataset
        return len(self.y)

    def __getitem__(self, idx):
        # gets a data point from the dataset as torch tensor array along with the label
        return self.X[idx], self.y[idx]


class TreeDataTwoPoints(Dataset):
    def __init__(self, dataset, number_of_points):
        self.dataset = dataset
        self.number_of_points = number_of_points
        self.original_number_of_points = dataset.shape[0]
        new_data = []
        new_labels = []
        for _ in range(self.number_of_points):
            idx = np.random.choice(self.original_number_of_points, 2, replace=False)
            first_point, first_label = self.dataset[idx[0]]
            second_point, second_label = self.dataset[idx[1]]
            if first_label >= second_label:
                new_labels.append(-1.0)
            else:
                new_labels.append(1.0)
            new_data.append(first_point.tolist() + second_point.tolist())
        self.X = torch.tensor(new_data).float()
        self.y = torch.tensor(new_labels).float()

    def __len__(self):
        # gets the number of rows in the dataset
        return len(self.y)

    def __getitem__(self, idx):
        # gets a data point from the dataset as torch tensor array along with the label
        return self.X[idx], self.y[idx]


class TorchDataHandler:
    def __init__(self, dataset, batch_size, num_classes=0, labels_names=None, transform=None, shuffle=False):
        if len(dataset) % batch_size != 0:
            raise ValueError("Length of dataset / batch size must be an integer.")
        self.curr_batch = 0
        self.num_classes = num_classes
        self.number_of_records = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = int(len(dataset)/batch_size)
        self.transform = transform
        self.original_dataset = {i: dataset[i][0] for i in range(self.number_of_records)}
        self.labels = {i: dataset[i][1] for i in range(self.number_of_records)}
        if not( self.transform ):
            self.transform = lambda val: val
        self.data = [(i, self.transform(dataset[i][0]), dataset[i][1]) for i in range(self.number_of_records)]
        if not( labels_names ):
            self.labels_names = {i: str(i) for i in range(self.num_classes)}
        else:
            self.labels_names = labels_names

    def get_num_classes(self):
        return self.num_classes

    def get_label_name(self, label_id):
        return self.labels_names[label_id]

    def get_num_batches(self):
        return self.num_batches

    def __len__(self):
        return self.number_of_records

    def get_original_sample(self, idx):
        return self.original_dataset[idx]

    def get_label(self, idx):
        return self.labels[idx]

    def get_record(self, idx):
        return self.transform(self.get_original_sample(idx))

    def __getitem__(self, idx):
        label = self.get_label(idx)
        original = self.get_original_sample(idx)
        record = self.get_record(idx)
        return {"id": idx, "label": label, "category": self.get_label_name(label),
                "original": original, "record": record}

    def enable_shuffle(self):
        self.shuffle = True

    def disable_shuffle(self):
        self.shuffle = False

    def next_batch(self):
        if self.curr_batch == 0 & self.shuffle:
            random.shuffle(self.data)
        start_ind = self.curr_batch * self.batch_size  # included
        end_ind = start_ind + self.batch_size  # not included
        curr_data = self.data[start_ind:end_ind]
        self.curr_batch = (self.curr_batch+1) % self.num_batches
        n = []
        ids = []
        objs = []
        labels = []
        for h in curr_data:
            ids.append(h[0])
            objs.append(h[1].tolist())
            labels.append(h[2])
        ids = torch.tensor(ids)
        objs = torch.tensor(objs)
        labels = torch.tensor(labels)
        n.append(ids)
        n.append(objs)
        n.append(labels)
        return n

    def all_batches(self):
        if self.curr_batch != 0:
            raise ValueError('The method all_batches can be executed if and only if the current batch begins with the '
                             'first one.')
        n = []
        for i in range(self.get_num_batches()):
            n.append(self.next_batch())
        return n


# ==============================================================================================================
# MODEL PERFORMANCE
# ==============================================================================================================


def model_accuracy(confusion_matrix):
    N = confusion_matrix.shape[0]
    acc = sum([confusion_matrix[i, i] for i in range(N)])
    class_performance = {i: {} for i in range(N)}
    for i in range(N):
        positive_rate = confusion_matrix[i, i]
        true_positives = confusion_matrix[i, :].sum()
        predicted_positives = confusion_matrix[:, i].sum()
        class_performance[i]["precision"] = positive_rate/predicted_positives if predicted_positives != 0 else 0
        class_performance[i]["recall"] = positive_rate/true_positives if true_positives != 0 else 0
        class_performance[i]["f1"] = (2*class_performance[i]["precision"]*class_performance[i]["recall"])/(class_performance[i]["precision"]+class_performance[i]["recall"]) if class_performance[i]["precision"]+class_performance[i]["recall"] != 0 else 0
    return acc/confusion_matrix.sum(), class_performance


def model_cost(confusion_matrix, cost_matrix):
    N = confusion_matrix.shape[0]
    a = np.subtract(cost_matrix, np.identity(N))
    return np.multiply(confusion_matrix, a).sum()


def spearman_footrule(y_true, y_pred):
    y_true = np.argsort(y_true, kind="heapsort")[::-1]
    y_pred = np.argsort(y_pred, kind="heapsort")[::-1]
    r = y_true.shape[0]
    if r % 2 == 0:
        r = r ** 2
    else:
        r = r ** 2 - 1
    return (3.0/float(r))*np.absolute(np.subtract(y_true, y_pred)).sum()


# ==============================================================================================================
# LOSS
# ==============================================================================================================


def two_points_compare_loss(first_pred, second_pred, feedback):
    '''
    feedback is a 1-D tensor containing either 1 or -1. Length of feedback is equal to the number of records of the feedback data.
    first_pred and second_pred are both scalar tensor resulting from the application of a neural network model
    to some input. The computed formula is:
    feedback*(first_pred - second_pred)
    therefore, if feedback is -1 then the user says that first error is more serious than the second one.
    in this way if the models says that first_pred is greater then the loss decreases because the model is correct,
    otherwise the loss increases.
    if feedback is 1 then the user says that second error is more serious than the first one.
    '''
    return torch.sum( feedback*(first_pred - second_pred) )


# ==============================================================================================================
# TRAINER
# ==============================================================================================================


class Trainer(ABC):
    def __init__(self, net, device, data):
        self.net = net
        self.device = device
        self.data = data

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
        cf_matrix = confusion_matrix(y_true, y_pred)
        return model_accuracy(cf_matrix)

    def evaluate_regressor(self, dataloader):
        total, correct = 0, 0
        y_true = []
        y_pred = []
        self.net.eval()
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            outputs = self.net(inputs)
            total += labels.size(0)
            y_pred.extend(outputs.tolist())
            y_true.extend(labels.tolist())
        return r2_score(y_true, y_pred)

    def predict(self, X):
        self.net.eval()
        X = X.to(self.device)
        return self.net(X)


class StandardBatchTrainer(Trainer):
    def __init__(self, net, device, dataloader, loss_fn, optimizer_name='adam',
                 verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, max_epochs=20):
        super(StandardBatchTrainer, self).__init__(net, device, dataloader)
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
        loss_arr = []
        loss_epoch_arr = []
        self.net.train()
        for epoch in range(self.max_epochs):
            for batch in self.data:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_arr.append(loss.item())
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_arr, loss_epoch_arr


class TwoPointsCompareTrainer(Trainer):
    def __init__(self, net, device, dataloader, optimizer_name='adam',
                 verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, max_epochs=20):
        super(TwoPointsCompareTrainer, self).__init__(net, device, dataloader)
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
        loss_arr = []
        loss_epoch_arr = []
        self.net.train()
        for epoch in range(self.max_epochs):
            for batch in self.data:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
                single_point_dim = inputs.shape[1]//2
                inputs_1 = inputs[:, :single_point_dim]
                inputs_2 = inputs[:, single_point_dim:]
                optimizer.zero_grad()
                outputs_1, outputs_2 = self.net(inputs_1), self.net(inputs_1)
                loss = torch.sum(labels * (outputs_1 - outputs_2))
                loss.backward()
                optimizer.step()
                loss_arr.append(loss.item())
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_arr, loss_epoch_arr

    def evaluate_regressor(self, dataloader):
        total, correct = 0, 0
        y_true = []
        y_pred = []
        self.net.eval()
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            outputs = self.net(inputs)
            total += labels.size(0)
            y_pred.extend(outputs.tolist())
            y_true.extend(labels.tolist())
        return spearman_footrule(y_true, y_pred)

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
    def __init__(self, activation_func, final_activation_func, input_layer_size: int, output_layer_size: int, hidden_layer_sizes: List[int] = []):
        super(MLPNet, self).__init__()
        self.activation_func = activation_func  # e.g., nn.ReLU()
        self.final_activation_func = final_activation_func  # e.g., nn.Tanh()
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size

        linear_layers = []
        layer_sizes = hidden_layer_sizes + [output_layer_size]
        curr_dim = input_layer_size
        for i in range(len(layer_sizes)):
            linear_layers.append(nn.Linear(curr_dim, layer_sizes[i]))
            if i != len(layer_sizes) - 1:
                linear_layers.append(self.activation_func)
            else:
                linear_layers.append(self.final_activation_func)
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
