import torch.nn as nn

from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator
from deeplearn.comparator.NeuralNetComparatorFactory import NeuralNetComparatorFactory
from deeplearn.comparator.OneOutputNeuronsComparator import OneOutputNeuronsComparator


class OneOutputNeuronsComparatorFactory(NeuralNetComparatorFactory):

    def create(self, net: nn.Module) -> NeuralNetComparator:
        return OneOutputNeuronsComparator(net)
