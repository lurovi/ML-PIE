from abc import abstractmethod, ABC
from typing import Set
from genepro.node import Node
from nsgp.sampling.PairChooser import PairChooser


class PairChooserFactory(ABC):

    @abstractmethod
    def create(self, n_pairs: int = 1, already_seen: Set[Node] = None) -> PairChooser:
        pass
