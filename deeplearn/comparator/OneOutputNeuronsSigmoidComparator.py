import torch.nn as nn
from typing import Any
import torch
from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator


class OneOutputNeuronsSigmoidComparator(NeuralNetComparator):
    def __init__(self, net: nn.Module):
        super(OneOutputNeuronsSigmoidComparator, self).__init__(net)

    def compare(self, point_1: Any, point_2: Any) -> bool:
        point_1, point_2 = point_1[0], point_2[0]
        point = torch.cat((point_1, point_2), dim=0).float().reshape(1, -1)
        output, _ = self.apply(point)
        output = output[0][0].item()
        return output < 0.5  # the neural network predicted class 0, it means that first element is lower than the second one
