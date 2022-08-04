import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torch.optim as optim
from deeplearn.dataset.TreeData import TreeData
from deeplearn.dataset.TreeDataTwoPointsCompare import TreeDataTwoPointsCompare
from deeplearn.trainer.Trainer import Trainer
from util.EvaluationMetrics import EvaluationMetrics
from util.Sort import Sort


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
        ddd = TreeData(None, points, y_true, scaler=None)
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
        return EvaluationMetrics.model_accuracy(cf_matrix)

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
        return EvaluationMetrics.spearman_footrule(y_true, y_pred, lambda x, y: torch.equal(x[0], y[0]))
