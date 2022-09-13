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

    @staticmethod
    def uncertainty_L2_sampler_online(X: torch.Tensor, y: torch.Tensor, trainer: Trainer, already_seen: List[int]) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        _, uncertainty, embeddings = trainer.predict(X)
        _, ind_points = Sort.heapsort(uncertainty, lambda a, b: a < b, inplace=False, reverse=True)
        count = 0
        i = 0
        points = []
        while count < 1 and i < len(ind_points):
            if ind_points[i] not in already_seen:
                already_seen.append(ind_points[i])
                count += 1
                points.append((X[ind_points[i]], y[ind_points[i]].item()))
            i += 1
        first_point, first_label = points[0]
        first_emb = embeddings[ind_points[i]]
        count = 0
        i = 0
        #euclidean_distances = torch.sum((first_emb.reshape() - embeddings)**2, axis=0)
        second_point, second_label = points[1]
        if first_label >= second_label:
            curr_feedback = np.array([-1], dtype=np.float32)
        else:
            curr_feedback = np.array([1], dtype=np.float32)
        curr_point = np.array([first_point.tolist() + second_point.tolist()], dtype=np.float32)
        return curr_point, curr_feedback, already_seen
