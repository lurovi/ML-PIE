from gp.tree.PrimitiveSet import PrimitiveSet
from gp.tree.PrimitiveTree import PrimitiveTree
import random
from gp.tree.TerminalSet import TerminalSet
from gp.tree.TreeGenerator import TreeGenerator


class FullGenerator(TreeGenerator):
    def __init__(self, primitive_set: PrimitiveSet, terminal_set: TerminalSet, min_height: int, max_height: int):
        self.__primitive_set = primitive_set
        self.__terminal_set = terminal_set
        self.__min_height = min_height
        self.__max_height = max_height

    def generate_tree(self) -> PrimitiveTree:
        primitive_set, terminal_set, min_height, max_height = self.__primitive_set, self.__terminal_set, self.__min_height, self.__max_height
        max_degree = primitive_set.max_arity()
        if not (min_height <= max_height):
            raise AttributeError("Min height must be less than or equal to max height.")
        if min_height < 1:
            raise AttributeError("Min height must be a positive number of at least 1.")
        if min_height == 1 and not (terminal_set.is_there_type(primitive_set.return_type())):
            min_height += 1
            if min_height > max_height:
                max_height += 1
        height = random.randint(min_height, max_height)
        if height != 1:
            tree = [[primitive_set.sample_root().name()]]
        else:
            tree = [[terminal_set.sample_typed(primitive_set.return_type())]]
        for layer_ind in range(1, height):
            curr_layer = [""] * (max_degree ** layer_ind)
            previous_layer = tree[layer_ind - 1]
            parents = [(iii, primitive_set.get_primitive(previous_layer[iii])) for iii in range(len(previous_layer)) if
                       previous_layer[iii] != ""]
            for parent_ind, parent_pr in parents:
                start_ind = parent_ind * max_degree
                for t in parent_pr.parameter_types():
                    if layer_ind == height - 1:
                        curr_layer[start_ind] = terminal_set.sample_typed(t)
                    else:
                        curr_layer[start_ind] = primitive_set.sample_typed(t).name()
                    start_ind += 1
            tree.append(curr_layer)
        for layer_ind in range(height, max_height):
            tree.append([""] * (max_degree ** layer_ind))
        return PrimitiveTree(tree, primitive_set, terminal_set)
