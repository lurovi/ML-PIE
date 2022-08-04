from abc import ABC, abstractmethod
import torch.nn as nn

from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator


class NeuralNetComparatorFactory(ABC):

    @abstractmethod
    def create(self, net: nn.Module) -> NeuralNetComparator:
        pass
