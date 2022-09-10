from abc import ABC, abstractmethod

from genepro.node import Node


class GroundTruthComputer(ABC):

    @abstractmethod
    def compute(self, tree: Node) -> float:
        pass
