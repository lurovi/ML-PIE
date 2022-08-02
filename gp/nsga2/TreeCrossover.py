import numpy as np
from pymoo.core.crossover import Crossover

from gp.operator import Crossover as TCrossover


class TreeCrossover(Crossover):
    def __init__(self, tree_crossover: TCrossover):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.tree_crossover = tree_crossover

    def _do(self, problem, x, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = x.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        y = np.full_like(x, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            a, b = x[0, k, 0], x[1, k, 0]

            # prepare the offsprings
            offsprings = self.tree_crossover.cross([a, b])
            y[0, k, 0], y[1, k, 0] = offsprings[0], offsprings[1]

        return y
