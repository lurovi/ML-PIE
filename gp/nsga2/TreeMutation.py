from pymoo.core.mutation import Mutation

from gp.operator import Mutation as TMutation


class TreeMutation(Mutation):

    def __init__(self, tree_mutation: TMutation):
        super().__init__()
        self.tree_mutation: TMutation
        self.tree_mutation = tree_mutation

    def _do(self, problem, x, **kwargs):
        # for each individual
        for i in range(len(x)):
            x[i, 0] = self.tree_mutation.mutate(x[i, 0])

        return x
