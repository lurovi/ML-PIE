from pymoo.core.mutation import Mutation

from nsgp.structure.TreeStructure import TreeStructure


class TreeMutation(Mutation):

    def __init__(self, structure: TreeStructure):
        super().__init__()
        self.__structure: TreeStructure = structure

    def _do(self, problem, x, **kwargs):
        # for each individual
        for i in range(len(x)):
            x[i, 0] = self.__structure.safe_subtree_mutation(x[i, 0])
        return x
