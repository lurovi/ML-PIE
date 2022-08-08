import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from deeplearn.dataset.TreeData import TreeData
from deeplearn.dataset.TreeDataTwoPointsCompare import TreeDataTwoPointsCompare
from deeplearn.trainer.Trainer import Trainer
from util.EvaluationMetrics import EvaluationMetrics
from util.Sort import Sort


class OnlineTwoPointsCompareDoubleInputTrainer(Trainer):
    def __init__(self, net, device, comparator_factory, loss_fn, data=None, optimizer_name='adam',
                 is_classification_task=False, verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0):
        super(OnlineTwoPointsCompareDoubleInputTrainer, self).__init__(net, device, data, 1)
        if data is not None and len(data) != 1:
            raise AttributeError("Online training requires a training set with exactly one record at time.")
        self.is_classification_task = is_classification_task
        self.comparator_factory = comparator_factory
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.output_layer_size = net.number_of_output_neurons()
        self.input_layer_size = net.number_of_input_neurons()
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.net_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.net_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                                  momentum=self.momentum, dampening=self.dampening)
        else:
            raise ValueError(f"{self.optimizer_name} is not a valid value for argument optimizer.")

    def change_data(self, data: Dataset) -> None:
        if len(data) != 1:
            raise AttributeError("Online training requires a training set with exactly one record at time.")
        super().change_data(data)

    def train(self):
        loss_epoch_arr = []
        self.set_train_mode()
        inputs, labels = self.all_batches()[0]
        inputs = self.to_device(inputs).float()
        if self.is_classification_task:
            labels = self.to_device(labels).long()
        else:
            labels = self.to_device(labels).float().reshape((labels.shape[0], 1))
        self.optimizer.zero_grad()
        outputs, _ = self.apply(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        loss_epoch_arr.append(loss.item())
        if self.verbose:
            print(f"Loss: {loss.item()}.")
        return loss_epoch_arr

    def evaluate_classifier(self, dataloader):
        y_true = []
        points = []
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = self.to_device(inputs).float(), self.to_device(labels).float().reshape((labels.shape[0], 1))
            for i in range(len(inputs)):
                points.append(inputs[i].tolist())
                y_true.append(labels[i][0].item())
        y_true = np.array(y_true, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        ddd = TreeData(None, points, y_true, scaler=None)
        ddd = TreeDataTwoPointsCompare(ddd, 2000, binary_label=True)
        ddd = DataLoader(ddd, batch_size=1, shuffle=True)
        y_true = []
        y_pred = []
        self.set_eval_mode()
        with torch.no_grad():
            for batch in ddd:
                inputs, labels = batch
                inputs, labels = self.to_device(inputs).float(), self.to_device(labels).float()
                outputs, _ = self.apply(inputs)
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
        self.set_train_mode()
        return EvaluationMetrics.model_accuracy(cf_matrix)

    def evaluate_ranking(self, dataloader):
        points = []
        self.set_eval_mode()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = self.to_device(inputs).float(), self.to_device(labels).float().reshape((labels.shape[0], 1))
                for i in range(len(inputs)):
                    points.append((inputs[i], labels[i][0].item()))
            y_true, _ = Sort.heapsort(points, lambda x, y: x[1] < y[1], inplace=False, reverse=False)
            comparator = self.create_comparator(self.comparator_factory)
            y_pred, _ = Sort.heapsort(points, comparator.compare, inplace=False, reverse=False)
        self.set_train_mode()
        return EvaluationMetrics.spearman_footrule(y_true, y_pred, lambda x, y: torch.equal(x[0], y[0]))
