import random
import torch

from typing import List, Tuple
import numpy as np

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.trainer.Trainer import Trainer
from util.Sort import Sort

from abc import ABC, abstractmethod

from copy import deepcopy


class PairSampler(ABC):
    def __init__(self, n_pairs: int = 20, already_seen: List[int] = None):
        self.__n_pairs = n_pairs
        if already_seen is not None:
            self.__already_seen = deepcopy(already_seen)
        else:
            self.__already_seen = []

    def get_already_seen(self) -> List[int]:
        return deepcopy(self.__already_seen)

    def add_index_to_already_seen(self, idx: int) -> None:
        self.__already_seen.append(idx)

    def clear_already_seen(self) -> None:
        self.__already_seen = []

    def get_n_pairs(self) -> int:
        return self.__n_pairs

    def set_n_pairs(self, n_pairs: int) -> None:
        self.__n_pairs = n_pairs

    def index_in_already_seen(self, idx: int) -> bool:
        return idx in self.__already_seen

    @abstractmethod
    def sample(self, X: torch.Tensor, y: torch.Tensor, trainer: Trainer = None) -> NumericalData:
        pass

    @abstractmethod
    def get_string_repr(self) -> str:
        pass
