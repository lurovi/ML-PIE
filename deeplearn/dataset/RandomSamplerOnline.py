import random
import torch

from typing import List
import numpy as np

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.trainer.Trainer import Trainer


class RandomSamplerOnline(PairSampler):
    def __init__(self, n_pairs: int = 20, already_seen: List[int] = None):
        super().__init__(1, already_seen)

    def sample(self, X: torch.Tensor, y: torch.Tensor, trainer: Trainer = None) -> NumericalData:
        idx_1, first_point, first_label, second_point, second_label = None, None, None, None, None
        train_indexes = list(range(len(y)))
        exit_loop = False
        while not (exit_loop):
            idx_1 = random.choice(train_indexes)
            if not self.index_in_already_seen(idx_1):
                exit_loop = True
                self.add_index_to_already_seen(idx_1)
                first_point, first_label = X[idx_1], y[idx_1].item()
        exit_loop = False
        while not (exit_loop):
            idx_2 = random.choice(train_indexes)
            if idx_2 != idx_1 and not self.index_in_already_seen(idx_2):
                exit_loop = True
                self.add_index_to_already_seen(idx_2)
                second_point, second_label = X[idx_2], y[idx_2].item()
        if first_label >= second_label:
            curr_feedback = np.array([-1], dtype=np.float32)
        else:
            curr_feedback = np.array([1], dtype=np.float32)
        curr_point = np.array([first_point.tolist() + second_point.tolist()], dtype=np.float32)
        return NumericalData(curr_point, curr_feedback)

    def get_string_repr(self) -> str:
        return "Random Sampler Online"
