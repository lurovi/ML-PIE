import torch.nn as nn

from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator
from deeplearn.comparator.NeuralNetComparatorFactory import NeuralNetComparatorFactory
from deeplearn.comparator.TwoOutputNeuronsSoftmaxComparator import TwoOutputNeuronsSoftmaxComparator


class TwoOutputNeuronsSoftmaxComparatorFactory(NeuralNetComparatorFactory):

    def create(self, net: nn.Module) -> NeuralNetComparator:
        return TwoOutputNeuronsSoftmaxComparator(net)
