import random
import torch

from typing import List
import numpy as np

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.trainer.Trainer import Trainer
from util.Sort import Sort


class UncertaintySamplerOnline(PairSampler):
    def __init__(self, n_pairs: int = 20, already_seen: List[int] = None):
        super().__init__(1, already_seen)

    def sample(self, X: torch.Tensor, y: torch.Tensor, trainer: Trainer = None) -> NumericalData:
        _, uncertainty, _ = trainer.predict(X)
        _, ind_points = Sort.heapsort(uncertainty, lambda a, b: a < b, inplace=False, reverse=True)
        count = 0
        i = 0
        points = []
        while count < 2 and i < len(ind_points):
            if not self.index_in_already_seen(ind_points[i]):
                self.add_index_to_already_seen(ind_points[i])
                count += 1
                points.append((X[ind_points[i]], y[ind_points[i]].item()))
            i += 1
        first_point, first_label = points[0]
        second_point, second_label = points[1]
        if first_label >= second_label:
            curr_feedback = np.array([-1], dtype=np.float32)
        else:
            curr_feedback = np.array([1], dtype=np.float32)
        curr_point = np.array([first_point.tolist() + second_point.tolist()], dtype=np.float32)
        return NumericalData(curr_point, curr_feedback)
