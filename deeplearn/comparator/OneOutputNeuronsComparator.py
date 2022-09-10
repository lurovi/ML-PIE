import torch.nn as nn
from typing import Any

from deeplearn.comparator.NeuralNetComparator import NeuralNetComparator


class OneOutputNeuronsComparator(NeuralNetComparator):
    def __init__(self, net: nn.Module):
        super().__init__(net=net)

    def compare(self, point_1: Any, point_2: Any) -> bool:
        point_1, point_2 = point_1[0], point_2[0]
        output_1, _ = self.apply(point_1.reshape(1, -1))
        output_1 = output_1[0][0].item()
        output_2, _ = self.apply(point_2.reshape(1, -1))
        output_2 = output_2[0][0].item()
        return output_1 < output_2  # first element is lower than the second one
