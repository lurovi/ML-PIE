import torch.nn as nn

from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator
from deeplearn.comparator.NeuralNetComparatorFactory import NeuralNetComparatorFactory
from deeplearn.comparator.TwoOutputNeuronsComparator import TwoOutputNeuronsComparator


class TwoOutputNeuronsComparatorFactory(NeuralNetComparatorFactory):

    def create(self, net: nn.Module) -> NeuralNetComparator:
        return TwoOutputNeuronsComparator(net)
