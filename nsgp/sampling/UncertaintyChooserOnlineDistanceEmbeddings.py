import random
import torch

from typing import Set, Tuple, List
import numpy as np
from genepro.node import Node

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.trainer.Trainer import Trainer
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.sampling.PairChooser import PairChooser
from util.Sort import Sort


class UncertaintyChooserOnlineDistanceEmbeddings(PairChooser):
    def __init__(self, n_pairs: int = 1, already_seen: Set[Node] = None, normalization_func: str = "max", lambda_coeff: float = 0.5):
        super().__init__(1, already_seen)
        self.__lambda_coeff = lambda_coeff
        self.__normalization_func_name = normalization_func
        if normalization_func not in ("max", "median", "mean"):
            raise AttributeError(
                f"{normalization_func} is not a valid normalization function for uncertainty sampler with distance embeddings.")
        if normalization_func == "max":
            self.__normalization_func = np.max
        elif normalization_func == "mean":
            self.__normalization_func = np.mean
        elif normalization_func == "median":
            self.__normalization_func = np.median

    def sample(self, queue: Set[Node], encoder: TreeEncoder = None, trainer: Trainer = None) -> List[Tuple[Node, Node]]:
        curr_queue = list(queue)
        curr_encodings = torch.from_numpy(np.array([encoder.encode(t, True) for t in curr_queue])).float()
        candidates = []
        already_seen_indexes = []
        _, uncertainty, embeddings = trainer.predict(curr_encodings)
        _, ind_points = Sort.heapsort(uncertainty, lambda a, b: a < b, inplace=False, reverse=True)
        count = 0
        i = 0
        first_emb = None
        while count < 1 and i < len(ind_points):
            idx = ind_points[i]
            curr_tree = curr_queue[idx]
            if idx not in already_seen_indexes:
                already_seen_indexes.append(idx)
                if not self.node_in_already_seen(curr_tree):
                    self.add_node_to_already_seen(curr_tree)
                    count += 1
                    candidates.append(curr_tree)
                    first_emb = embeddings[idx].repeat(len(ind_points), 1)
            i += 1
        dist = ((first_emb - embeddings) ** 2).sum(axis=1).tolist()
        scaling_uncert, scaling_dist = self.__normalization_func(uncertainty), self.__normalization_func(dist)
        l2_uncertainty = [uncertainty[i] / float(scaling_uncert) + self.__lambda_coeff * (dist[i] / float(scaling_dist))
                          for i in range(len(uncertainty))]
        _, ind_points = Sort.heapsort(l2_uncertainty, lambda a, b: a < b, inplace=False, reverse=True)
        count = 0
        i = 0
        while count < 1 and i < len(ind_points):
            idx = ind_points[i]
            curr_tree = curr_queue[idx]
            if idx not in already_seen_indexes:
                already_seen_indexes.append(idx)
                if not self.node_in_already_seen(curr_tree):
                    self.add_node_to_already_seen(curr_tree)
                    count += 1
                    candidates.append(curr_tree)
            i += 1
        for first_tree in candidates:
            queue.remove(first_tree)
        return [(candidates[0], candidates[1])]

    def get_string_repr(self) -> str:
        return "Uncertainty Sampler Online L2 Distances " + (self.__normalization_func_name[0].upper() + self.__normalization_func_name[1:]) + " Norm"
