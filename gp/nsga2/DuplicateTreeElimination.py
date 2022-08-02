from typing import List, Any

from pymoo.core.duplicate import ElementwiseDuplicateElimination


class DuplicateTreeElimination(ElementwiseDuplicateElimination):
    #def __init__(self, data: List[List[Any]]):
    #    super(DuplicateTreeElimination, self).__init__()
    #    self.__data = data

    # TODO provide actual implementation
    def is_equal(self, a, b):
        a_tree, b_tree = a.X[0], b.X[0]
        '''
        a_eval, b_eval = [], []
        for l in self.__data:
            a_eval.append(a_tree.compile(l))
            b_eval.append(b_tree.compile(l))
        return all([True if a_eval[i] == b_eval[i] else False for i in range(len(self.__data))])
        '''
        return a_tree == b_tree
