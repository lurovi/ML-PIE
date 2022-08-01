from abc import abstractmethod, ABC
from typing import Tuple, List
from gp.tree.PrimitiveTree import PrimitiveTree
from gp.evolution.Population import Population


class Selection(ABC):

    @abstractmethod
    def select(self, population: Population) -> Tuple[List[PrimitiveTree], List[List[float]]]:
        pass
