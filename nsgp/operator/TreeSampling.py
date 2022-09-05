from pymoo.core.sampling import Sampling
import numpy as np

from nsgp.util.TreeGrammarStructure import TreeGrammarStructure


class TreeSampling(Sampling):

    def __init__(self, structure: TreeGrammarStructure):
        super().__init__()
        self.__structure: TreeGrammarStructure = structure

    def _do(self, problem, n_samples, **kwargs):
        x = np.empty((n_samples, 1), dtype=object)

        for i in range(n_samples):
            x[i, 0] = self.__structure.generate_tree()

        return x
