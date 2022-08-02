from abc import abstractmethod, ABC
from typing import List
from gp.tree.PrimitiveTree import PrimitiveTree


class Crossover(ABC):

    @abstractmethod
    def mate(self, individuals: List[PrimitiveTree]) -> List[PrimitiveTree]:
        pass
