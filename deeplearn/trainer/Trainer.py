from typing import Tuple, List, Any, Iterator

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from torch import optim
from torch.utils.data import Dataset, DataLoader

from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator
from deeplearn.comparator.NeuralNetComparatorFactory import NeuralNetComparatorFactory


class Trainer(ABC):
    def __init__(self, net: nn.Module,
                 device: torch.device,
                 data: Dataset,
                 optimizer_name: str,
                 batch_size: int = 1,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.00001,
                 momentum: float = 0,
                 dampening: float = 0,
                 custom_optimizer: torch.optim.Optimizer = None):
        self.__device: torch.device = device
        self.__batch_size: int = batch_size
        self.__net: nn.Module = net.to(self.__device)
        self.__data: Dataset = data
        if self.__data is not None:
            self.__dataloader: DataLoader = DataLoader(self.__data, batch_size=self.__batch_size, shuffle=True)
        self.__learning_rate: float = learning_rate
        self.__weight_decay: float = weight_decay
        self.__momentum: float = momentum
        self.__dampening: float = dampening
        self.__optimizer_name: str = optimizer_name
        if custom_optimizer is not None:
            self.__optimizer: torch.optim.Optimizer = custom_optimizer
        else:
            if self.__optimizer_name == 'adam':
                self.__optimizer: torch.optim.Optimizer = optim.Adam(self.net_parameters(), lr=self.get_learning_rate(), weight_decay=self.get_weight_decay())
            elif self.__optimizer_name == 'sgd':
                self.__optimizer: torch.optim.Optimizer = optim.SGD(self.net_parameters(), lr=self.get_learning_rate(), weight_decay=self.get_weight_decay(),
                                      momentum=self.get_momentum(), dampening=self.get_dampening())
            else:
                raise ValueError(f"{self.__optimizer_name} is not a valid value for argument optimizer.")
        self.__output_layer_size: int = net.number_of_output_neurons()
        self.__input_layer_size: int = net.number_of_input_neurons()

    def get_output_layer_size(self) -> int:
        return self.__output_layer_size

    def get_input_layer_size(self) -> int:
        return self.__input_layer_size

    def get_learning_rate(self) -> float:
        return self.__learning_rate

    def get_weight_decay(self) -> float:
        return self.__weight_decay

    def get_momentum(self) -> float:
        return self.__momentum

    def get_dampening(self) -> float:
        return self.__dampening

    def get_net(self) -> nn.Module:
        return self.__net

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    def optimizer_zero_grad(self) -> None:
        self.__optimizer.zero_grad()

    def optimizer_step(self) -> None:
        self.__optimizer.step()

    def net_zero_grad(self) -> None:
        self.__net.zero_grad()

    def get_device(self) -> torch.device:
        return self.__device

    def set_train_mode(self) -> None:
        self.__net.train()

    def set_eval_mode(self) -> None:
        self.__net.eval()

    def net_parameters(self) -> Iterator[Any]:
        return self.__net.parameters()

    def apply(self, X: torch.Tensor) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        return self.__net(X)

    def to_device(self, X: torch.Tensor) -> torch.Tensor:
        return X.to(self.__device)

    def all_batches(self) -> List[Any]:
        return [b for b in self.__dataloader]

    def number_of_points(self) -> int:
        return len(self.__data)

    def get_point(self, idx: int) -> Any:
        n_points: int = self.number_of_points()
        if not (0 <= idx < n_points):
            raise IndexError(f"{idx} is out of range for the training set of size {n_points}.")
        return self.__data[idx]

    def change_data(self, data: Dataset) -> None:
        self.__data = data
        if self.__data is not None:
            self.__dataloader = DataLoader(self.__data, batch_size=self.__batch_size, shuffle=True)

    def get_batch_size(self) -> int:
        return self.__batch_size

    def set_batch_size(self, batch_size: int) -> None:
        self.__batch_size = batch_size

    def create_comparator(self, comparator_factory: NeuralNetComparatorFactory) -> NeuralNetComparator:
        return comparator_factory.create(self.__net)

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        self.set_eval_mode()
        with torch.no_grad():
            X: torch.Tensor = self.to_device(X)
            res: Tuple[torch.Tensor, List[float], torch.Tensor] = self.apply(X)
        self.set_train_mode()
        return res

    @abstractmethod
    def fit(self) -> List[float]:
        pass
