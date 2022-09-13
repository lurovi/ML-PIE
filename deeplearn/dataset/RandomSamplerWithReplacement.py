import random
import torch

from typing import List
import numpy as np

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.trainer.Trainer import Trainer


class RandomSamplerWithReplacement(PairSampler):
    def __init__(self, n_pairs: int = 20, already_seen: List[int] = None):
        super().__init__(n_pairs, None)

    def sample(self, X: torch.Tensor, y: torch.Tensor, trainer: Trainer = None) -> NumericalData:
        second_label, second_point = None, None
        train_indexes = list(range(len(y)))
        X_pairs, y_pairs = [], []
        for _ in range(self.get_n_pairs()):
            idx_1 = random.choice(train_indexes)
            first_point, first_label = X[idx_1], y[idx_1].item()
            exit_loop = False
            while not (exit_loop):
                idx_2 = random.choice(train_indexes)
                if idx_2 != idx_1:
                    exit_loop = True
                    second_point, second_label = X[idx_2], y[idx_2].item()
            if first_label >= second_label:
                curr_feedback = -1
            else:
                curr_feedback = 1
            curr_point = first_point.tolist() + second_point.tolist()
            y_pairs.append(curr_feedback)
            X_pairs.append(curr_point)
        return NumericalData(np.array(X_pairs, dtype=np.float32), np.array(y_pairs, dtype=np.float32))
