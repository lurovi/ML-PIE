from abc import ABC, abstractmethod

from genepro.node import Node


class GroundTruthComputer(ABC):
    def __init__(self):
        self.__name: str = None

    def get_name(self) -> str:
        return self.__name

    def set_name(self, name: str) -> None:
        if self.__name is None:
            self.__name = name
        else:
            raise AttributeError(f"Once the name has already been set, it can't be modified anymore.")

    @abstractmethod
    def compute(self, tree: Node) -> float:
        pass
