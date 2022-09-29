import random
import threading

import torch

from typing import Set, Tuple, List
import numpy as np
from genepro.node import Node

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.trainer.Trainer import Trainer
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.sampling.PairChooser import PairChooser


class RandomChooser(PairChooser):
    def __init__(self, n_pairs: int = 1, already_seen: Set[Node] = None):
        super().__init__(n_pairs, already_seen)

    def sample(self, queue: Set[Node], encoder: TreeEncoder = None, trainer: Trainer = None, mutex: threading.Lock = None) -> List[Tuple[Node, Node]]:
        curr_queue = list(queue)
        train_indexes = list(range(len(curr_queue)))
        candidates = []
        already_seen_indexes = []
        for _ in range(self.get_n_pairs()):
            exit_loop = False
            while not (exit_loop):
                idx_1 = random.choice(train_indexes)
                first_tree = curr_queue[idx_1]
                if idx_1 not in already_seen_indexes:
                    already_seen_indexes.append(idx_1)
                    if not self.node_in_already_seen(first_tree):
                        self.add_node_to_already_seen(first_tree)
                        exit_loop = True
            exit_loop = False
            while not (exit_loop):
                idx_2 = random.choice(train_indexes)
                second_tree = curr_queue[idx_2]
                if idx_2 not in already_seen_indexes:
                    already_seen_indexes.append(idx_2)
                    if not self.node_in_already_seen(second_tree):
                        self.add_node_to_already_seen(second_tree)
                        exit_loop = True
            candidates.append((first_tree, second_tree))
        for first_tree, second_tree in candidates:
            queue.remove(first_tree)
            queue.remove(second_tree)
        return candidates

    def get_string_repr(self) -> str:
        return "Random Sampler"
