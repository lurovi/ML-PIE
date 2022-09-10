import torch
import torch.nn as nn
from typing import Any
from abc import ABC, abstractmethod


class NeuralNetComparator(ABC):
    def __init__(self, net: nn.Module):
        self.__net: nn.Module = net

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return self.__net(data)

    @abstractmethod
    def compare(self, point_1: Any, point_2: Any) -> bool:
        pass
