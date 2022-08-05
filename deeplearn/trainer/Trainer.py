from typing import Tuple, List, Any, Iterator

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score
from torch.utils.data import Dataset, DataLoader

from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator
from deeplearn.comparator.NeuralNetComparatorFactory import NeuralNetComparatorFactory
from util.EvaluationMetrics import EvaluationMetrics
from util.Sort import Sort


class Trainer(ABC):
    def __init__(self, net: nn.Module(), device: torch.device, data: Dataset, batch_size: int = 1):
        self.__device = device
        self.__batch_size = batch_size
        self.__net = net.to(self.__device)
        self.__data = data
        if self.__data is not None:
            self.__dataloader = DataLoader(self.__data, batch_size=self.__batch_size, shuffle=True)

    def set_train_mode(self) -> None:
        self.__net.train()

    def set_eval_mode(self) -> None:
        self.__net.eval()

    def net_parameters(self) -> Iterator[Any]:
        return self.__net.parameters()

    def apply(self, X: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        return self.__net(X)

    def to_device(self, X: torch.Tensor) -> torch.Tensor:
        return X.to(self.__device)

    def all_batches(self) -> List[Any]:
        return [b for b in self.__dataloader]

    def number_of_points(self) -> int:
        return len(self.__data)

    def get_point(self, idx: int) -> Any:
        n_points = self.number_of_points()
        if not (0 <= idx < n_points):
            raise IndexError(f"{idx} is out of range for the training set of size {n_points}.")
        return self.__data[idx]

    def change_data(self, data: Dataset) -> None:
        self.__data = data
        if self.__data is not None:
            self.__dataloader = DataLoader(self.__data, batch_size=self.__batch_size, shuffle=True)

    def set_batch_size(self, batch_size: int) -> None:
        self.__batch_size = batch_size

    def create_comparator(self, comparator_factory: NeuralNetComparatorFactory) -> NeuralNetComparator:
        return comparator_factory.create(self.__net)

    @abstractmethod
    def train(self) -> List[float]:
        pass

    def evaluate_classifier(self, dataloader):
        total, correct = 0, 0
        y_true = []
        y_pred = []
        self.set_eval_mode()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = self.to_device(inputs).float(), self.to_device(labels).float().reshape((labels.shape[0], 1))
                outputs, _ = self.apply(inputs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                y_pred.extend(pred.tolist())
                y_true.extend(labels.tolist())
            y_true = list(np.concatenate(y_true).flat)
            y_pred = list(np.concatenate(y_pred).flat)
            cf_matrix = confusion_matrix(y_true, y_pred)
        self.set_train_mode()
        return EvaluationMetrics.model_accuracy(cf_matrix)

    def evaluate_regressor(self, dataloader):
        y_true = []
        y_pred = []
        self.set_eval_mode()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = self.to_device(inputs).float(), self.to_device(labels).float().reshape((labels.shape[0], 1))
                outputs, _ = self.apply(inputs)
                y_pred.extend(outputs.tolist())
                y_true.extend(labels.tolist())
            y_true = list(np.concatenate(y_true).flat)
            y_pred = list(np.concatenate(y_pred).flat)
        self.set_train_mode()
        return r2_score(y_true, y_pred)

    def evaluate_ranking(self, dataloader):
        y_true = []
        y_pred = []
        self.set_eval_mode()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = self.to_device(inputs).float(), self.to_device(labels).float().reshape((labels.shape[0], 1))
                outputs, _ = self.apply(inputs)
                for i in range(len(inputs)):
                    curr_input, curr_label, curr_output = inputs[i], labels[i][0].item(), outputs[i][0].item()
                    y_true.append((curr_input, curr_label))
                    y_pred.append((curr_input, curr_output))
            y_true, _ = Sort.heapsort(y_true, lambda x, y: x[1] < y[1], inplace=False, reverse=False)
            y_pred, _ = Sort.heapsort(y_pred, lambda x, y: x[1] < y[1], inplace=False, reverse=False)
        self.set_train_mode()
        return EvaluationMetrics.spearman_footrule(y_true, y_pred, lambda x, y: torch.equal(x[0], y[0]))

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        self.set_eval_mode()
        with torch.no_grad():
            X = self.to_device(X)
            res = self.apply(X)
        self.set_train_mode()
        return res
