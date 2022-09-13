import torch

from typing import List
import numpy as np

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.trainer.Trainer import Trainer
from util.Sort import Sort


class UncertaintySamplerOnlineDistanceEmbeddings(PairSampler):
    def __init__(self, n_pairs: int = 20, already_seen: List[int] = None, normalization_func: str = "max", lambda_coeff: float = 0.5):
        super().__init__(1, already_seen)
        self.__lambda_coeff = lambda_coeff
        self.__normalization_func_name = normalization_func
        if normalization_func not in ("max", "median", "mean"):
            raise AttributeError(f"{normalization_func} is not a valid normalization function for uncertainty sampler with distance embeddings.")
        if normalization_func == "max":
            self.__normalization_func = np.max
        elif normalization_func == "mean":
            self.__normalization_func = np.mean
        elif normalization_func == "median":
            self.__normalization_func = np.median

    def sample(self, X: torch.Tensor, y: torch.Tensor, trainer: Trainer = None) -> NumericalData:
        _, uncertainty, embeddings = trainer.predict(X)
        _, ind_points = Sort.heapsort(uncertainty, lambda a, b: a < b, inplace=False, reverse=True)
        first_emb = None
        points = []
        count = 0
        i = 0
        while count < 1 and i < len(ind_points):
            if not self.index_in_already_seen(ind_points[i]):
                self.add_index_to_already_seen(ind_points[i])
                count += 1
                points.append((X[ind_points[i]], y[ind_points[i]].item()))
                first_emb = embeddings[ind_points[i]].repeat(len(ind_points), 1)
            i += 1
        first_point, first_label = points[0]
        dist = ((first_emb - embeddings)**2).sum(axis=1).tolist()
        scaling_uncert, scaling_dist = self.__normalization_func(uncertainty), self.__normalization_func(dist)
        l2_uncertainty = [uncertainty[i]/float(scaling_uncert) + self.__lambda_coeff * (dist[i]/float(scaling_dist)) for i in range(len(uncertainty))]
        _, ind_points = Sort.heapsort(l2_uncertainty, lambda a, b: a < b, inplace=False, reverse=True)
        count = 0
        i = 0
        while count < 1 and i < len(ind_points):
            if not self.index_in_already_seen(ind_points[i]):
                self.add_index_to_already_seen(ind_points[i])
                count += 1
                points.append((X[ind_points[i]], y[ind_points[i]].item()))
            i += 1
        second_point, second_label = points[1]
        if first_label >= second_label:
            curr_feedback = np.array([-1], dtype=np.float32)
        else:
            curr_feedback = np.array([1], dtype=np.float32)
        curr_point = np.array([first_point.tolist() + second_point.tolist()], dtype=np.float32)
        return NumericalData(curr_point, curr_feedback)

    def get_string_repr(self) -> str:
        return "Uncertainty Sampler Online L2 Distances "+(self.__normalization_func_name[0].upper() + self.__normalization_func_name[1:])+" Norm"
