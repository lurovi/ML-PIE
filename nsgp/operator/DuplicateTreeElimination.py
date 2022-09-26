from genepro.node import Node

from pymoo.core.duplicate import ElementwiseDuplicateElimination

import numpy as np


class DuplicateTreeElimination(ElementwiseDuplicateElimination):
    def __init__(self, little_data: np.ndarray = None):
        super().__init__()
        self.__little_data: np.ndarray = little_data

    def is_equal(self, a, b) -> bool:
        a_tree: Node = a.X[0]
        b_tree: Node = b.X[0]
        if self.__little_data is not None:
            return a_tree.semantically_equal(b_tree, self.__little_data)
        else:
            return a_tree.structurally_equal(b_tree)
