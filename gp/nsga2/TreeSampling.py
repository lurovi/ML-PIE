import numpy as np
from pymoo.core.sampling import Sampling

from gp.tree import TreeGenerator


class TreeSampling(Sampling):

    def __init__(self, tree_generator: TreeGenerator):
        super().__init__()
        self.tree_generator = tree_generator

    def _do(self, problem, n_samples, **kwargs):
        x = np.empty((n_samples, 1), dtype=object)

        for i in range(n_samples):
            x[i, 0] = self.tree_generator.generate_tree()

        return x
