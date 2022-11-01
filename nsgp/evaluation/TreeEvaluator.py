from abc import ABC, abstractmethod
from genepro.node import Node
import numpy as np


class TreeEvaluator(ABC):

    @abstractmethod
    def evaluate(self, tree: Node, **kwargs) -> float:
        pass
