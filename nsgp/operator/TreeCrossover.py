from pymoo.core.crossover import Crossover
import numpy as np

from nsgp.util.TreeGrammarStructure import TreeGrammarStructure


class TreeCrossover(Crossover):
    def __init__(self, structure: TreeGrammarStructure):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.__structure: TreeGrammarStructure = structure

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
            y[0, k, 0], y[1, k, 0] = self.__structure.safe_subtree_crossover_two_children(p1, p2)

        return y