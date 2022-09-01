from genepro.node import Node

from genepro.variation import generate_random_tree, safe_subtree_crossover_two_children, safe_subtree_mutation
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
import numpy as np


class DuplicateTreeElimination(ElementwiseDuplicateElimination):
    def __init__(self, little_data: np.ndarray):
        super().__init__()
        self.__little_data: np.ndarray = little_data

    def is_equal(self, a, b) -> bool:
        a_tree: Node = a.X[0]
        b_tree: Node = b.X[0]
        return a_tree.semantically_equals(b_tree, self.__little_data)
