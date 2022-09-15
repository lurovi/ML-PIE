from typing import Tuple, Set, List

from deeplearn.trainer.Trainer import Trainer
from genepro.node import Node

from abc import ABC, abstractmethod

from copy import deepcopy

from nsgp.encoder.TreeEncoder import TreeEncoder


class PairChooser(ABC):
    def __init__(self, n_pairs: int = 1, already_seen: Set[Node] = None):
        self.__n_pairs = n_pairs
        if already_seen is not None:
            self.__already_seen: Set[Node] = deepcopy(already_seen)
        else:
            self.__already_seen: Set[Node] = set()

    def get_already_seen(self) -> Set[Node]:
        return deepcopy(self.__already_seen)

    def add_node_to_already_seen(self, node: Node) -> None:
        self.__already_seen.add(node)

    def clear_already_seen(self) -> None:
        self.__already_seen = {}

    def get_n_pairs(self) -> int:
        return self.__n_pairs

    def set_n_pairs(self, n_pairs: int) -> None:
        self.__n_pairs = n_pairs

    def node_in_already_seen(self, node: Node) -> bool:
        return node in self.__already_seen

    def remove_node_from_already_seen(self, node: Node) -> None:
        self.__already_seen.remove(node)

    @abstractmethod
    def sample(self, queue: Set[Node], encoder: TreeEncoder = None, trainer: Trainer = None) -> List[Tuple[Node, Node]]:
        pass

    @abstractmethod
    def get_string_repr(self) -> str:
        pass
