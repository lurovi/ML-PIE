from gp.tree.PrimitiveSet import PrimitiveSet
from gp.tree.PrimitiveTree import PrimitiveTree
import random

from gp.tree.TerminalSet import TerminalSet
from gp.tree.TreeGenerator import TreeGenerator


class GrowGenerator(TreeGenerator):
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
        expand = [[True]]
        is_min_height_reached = True if min_height == 1 else False
        for layer_ind in range(1, height):
            curr_layer = [""] * (max_degree ** layer_ind)
            curr_expand = [False] * (max_degree ** layer_ind)
            if layer_ind + 1 == min_height:
                is_min_height_reached = True
            previous_layer = tree[layer_ind - 1]
            previous_expand = expand[layer_ind - 1]
            parents = [(iii, previous_expand[iii], primitive_set.get_primitive(previous_layer[iii])) for iii in
                       range(len(previous_layer)) if
                       previous_layer[iii] != "" and primitive_set.is_primitive(previous_layer[iii])]
            if not is_min_height_reached:
                to_expand_necessarily = random.randint(0, len(parents) - 1)
            else:
                to_expand_necessarily = -1
            for p_i in range(len(parents)):
                parent_ind = parents[p_i][0]
                parent_exp = parents[p_i][1]
                parent_pr = parents[p_i][2]
                start_ind = parent_ind * max_degree
                parameter_types = parent_pr.parameter_types()
                to_expand_necesserly_param = random.randint(0, len(parameter_types) - 1)
                for t_i in range(len(parameter_types)):
                    t = parameter_types[t_i]
                    if layer_ind == height - 1:
                        curr_layer[start_ind] = terminal_set.sample_typed(t)
                        curr_expand[start_ind] = False
                    else:
                        if parent_exp:
                            if random.random() <= 0.20:
                                curr_layer[start_ind] = terminal_set.sample_typed(t)
                                curr_expand[start_ind] = False
                            else:
                                curr_layer[start_ind] = primitive_set.sample_typed(t).name()
                                if random.random() < 0.50:
                                    curr_expand[start_ind] = True
                                else:
                                    curr_expand[start_ind] = False
                        elif to_expand_necessarily == p_i and to_expand_necesserly_param == t_i:
                            curr_layer[start_ind] = primitive_set.sample_typed(t).name()
                            curr_expand[start_ind] = True
                        else:
                            curr_layer[start_ind] = terminal_set.sample_typed(t)
                            curr_expand[start_ind] = False
                    start_ind += 1
            tree.append(curr_layer)
            expand.append(curr_expand)
        for layer_ind in range(height, max_height):
            tree.append([""] * (max_degree ** layer_ind))
        return PrimitiveTree(tree, primitive_set, terminal_set)
