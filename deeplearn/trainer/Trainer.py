import torch
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score

from util.EvaluationMetrics import EvaluationMetrics
from util.Sort import Sort


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
        return EvaluationMetrics.model_accuracy(cf_matrix)

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
        return EvaluationMetrics.spearman_footrule(y_true, y_pred, lambda x, y: torch.equal(x[0], y[0]))

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            X = X.to(self.device)
            res = self.net(X)
        self.net.train()
        return res
