import torch.nn as nn

from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator
from deeplearn.comparator.NeuralNetComparatorFactory import NeuralNetComparatorFactory
from deeplearn.comparator.OneOutputNeuronsSigmoidComparator import OneOutputNeuronsSigmoidComparator


class OneOutputNeuronsSigmoidComparatorFactory(NeuralNetComparatorFactory):

    def create(self, net: nn.Module) -> NeuralNetComparator:
        return OneOutputNeuronsSigmoidComparator(net)
