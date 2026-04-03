from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

from genepro.node import Node


class TreeEncoder(ABC):
    def __init__(self):
        self.__scaler: Any = None
        self.__name: str = None

    def get_scaler(self) -> Any:
        return self.__scaler

    def set_scaler(self, scaler: Any) -> None:
        if self.__scaler is None:
            self.__scaler = scaler
        else:
            raise AttributeError("Once the scaler has been set, it can't be modified anymore.")

    def get_name(self) -> str:
        return self.__name

    def set_name(self, name: str) -> None:
        if self.__name is None or self.__name.strip() == "":
            self.__name = name
        else:
            raise AttributeError("Once the name has been set, it can't be modified anymore.")

    def scale(self, encoding: np.ndarray) -> np.ndarray:
        if self.__scaler is None:
            raise ValueError("Current scaler of this tree encoder is None. It can't be used to scale anything at the moment.")
        if len(encoding.shape) == 1:
            return self.__scaler.transform(encoding.reshape(1, -1))[0]
        elif len(encoding.shape) == 2:
            return self.__scaler.transform(encoding)
        else:
            raise ValueError(f"The number of axis of encoding is {len(encoding.shape)}. However, it must be either 1 or 2.")

    @staticmethod
    def encoding_size(num_primitives: int, num_features: int, max_arity: int, max_n_levels: int) -> Tuple[
        int, int, int]:
        counts = num_primitives + num_features + 4
        level_wise_counts = max_n_levels * (num_primitives + num_features + 1) + 3
        one_hot = int((num_primitives + num_features + 1) * ((max_arity ** max_n_levels - 1) / float(max_arity - 1)))
        return counts, level_wise_counts, one_hot

    @abstractmethod
    def encode(self, tree: Node, apply_scaler: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def size(self) -> int:
        pass
