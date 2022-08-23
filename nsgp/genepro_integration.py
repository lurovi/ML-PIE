from genepro.variation import generate_random_tree, safe_subtree_crossover_two_children, safe_subtree_mutation
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
import numpy as np


class TreeSampling(Sampling):

    def __init__(self, internal_nodes: list, leaf_nodes: list, max_depth: int):
        super().__init__()
        self.internal_nodes = internal_nodes
        self.leaf_nodes = leaf_nodes
        self.max_depth = max_depth

    def _do(self, problem, n_samples, **kwargs):
        x = np.empty((n_samples, 1), dtype=object)

        for i in range(n_samples):
            x[i, 0] = generate_random_tree(self.internal_nodes, self.leaf_nodes, self.max_depth)

        return x


class TreeCrossover(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, x, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = x.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        y = np.full_like(x, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            p1, p2 = x[0, k, 0], x[1, k, 0]

            # prepare the offsprings
            y[0, k, 0], y[1, k, 0] = safe_subtree_crossover_two_children(p1, p2)

        return y


class TreeMutation(Mutation):

    def __init__(self, internal_nodes: list, leaf_nodes: list):
        super().__init__()
        self.internal_nodes = internal_nodes
        self.leaf_nodes = leaf_nodes

    def _do(self, problem, x, **kwargs):
        # for each individual
        for i in range(len(x)):
            x[i, 0] = safe_subtree_mutation(x[i, 0], self.internal_nodes, self.leaf_nodes)

        return x


class DuplicateTreeElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        a_tree, b_tree = a.X[0], b.X[0]
        return a_tree == b_tree
