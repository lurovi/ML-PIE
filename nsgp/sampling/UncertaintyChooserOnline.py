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


class UncertaintyChooserOnline(PairChooser):
    def __init__(self, n_pairs: int = 1, already_seen: Set[Node] = None):
        super().__init__(1, already_seen)

    def sample(self, queue: Set[Node], encoder: TreeEncoder = None, trainer: Trainer = None) -> List[Tuple[Node, Node]]:
        curr_queue = list(queue)
        curr_encodings = torch.from_numpy(np.array([encoder.encode(t, True) for t in curr_queue])).float()
        candidates = []
        already_seen_indexes = []
        _, uncertainty, _ = trainer.predict(curr_encodings)
        _, ind_points = Sort.heapsort(uncertainty, lambda a, b: a < b, inplace=False, reverse=True)
        count = 0
        i = 0
        while count < 2 and i < len(ind_points):
            idx = ind_points[i]
            curr_tree = curr_queue[idx]
            if idx not in already_seen_indexes:
                already_seen_indexes.append(idx)
                if not self.node_in_already_seen(curr_tree):
                    self.add_node_to_already_seen(curr_tree)
                    count += 1
                    candidates.append(curr_tree)
            i += 1

        if count < 2:
            if count == 0:
                candidates.append(curr_queue[ind_points[len(ind_points) - 1]])
                candidates.append(curr_queue[ind_points[0]])
                self.add_node_to_already_seen(candidates[0])
                self.add_node_to_already_seen(candidates[1])
            elif count == 1:
                candidates.append(curr_queue[ind_points[len(ind_points) - 1]])
                self.add_node_to_already_seen(candidates[1])

        for first_tree in candidates:
            queue.remove(first_tree)
        return [(candidates[0], candidates[1])]

    def get_string_repr(self) -> str:
        return "Uncertainty Sampler Online"
