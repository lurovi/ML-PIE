import random
import re
from abc import ABC, abstractmethod
import numpy as np

# ==============================================================================================================
# BUILDING BLOCKS
# ==============================================================================================================


class Constant:
    def __init__(self, name, val):
        self.__name = name
        self.__val = val
        self.__type = type(self.__val)

    def __str__(self):
        return f"{self.__name} : {self.__val} --> {self.__type}"

    def __call__(self):
        return self.__val

    def name(self):
        return self.__name

    def type(self):
        return self.__type

    def cast(self, val):
        return self.__type(val)


class Ephemeral:
    def __init__(self, name, func):
        self.__name = name
        self.__func = func
        self.__type = type(self.__func())

    def __str__(self):
        return f"{self.__name} : {self.__func} --> {self.__type}"

    def __call__(self):
        return self.__func()

    def name(self):
        return self.__name

    def type(self):
        return self.__type

    def cast(self, val):
        return self.__type(val)


class TerminalSet:
    def __init__(self, feature_types, constants, ephemeral):
        self.__num_features = len(feature_types)
        self.__num_constants = len(constants)
        self.__num_ephemeral = len(ephemeral)
        self.__feature_types = feature_types
        self.__constants = constants
        self.__ephemeral = ephemeral

        self.__all_obj = feature_types + self.__constants + self.__ephemeral
        self.__all_types = feature_types + [c.type() for c in self.__constants] + [e.type() for e in self.__ephemeral]
        self.__all_idx = ["x"+str(i) for i in range(self.__num_features)] + ["c"+str(i)+" "+str(self.__constants[i]()) for i in range(self.__num_constants)] + ["e"+str(i)+" " for i in range(self.__num_ephemeral)]

    def __str__(self):
        return f"N. Features: {self.__num_features} - N. Constants: {self.__num_constants} - N. Ephemeral: {self.__num_ephemeral}."

    def num_features(self):
        return self.__num_features

    def num_constants(self):
        return self.__num_constants

    def num_ephemeral(self):
        return self.__num_ephemeral

    def get_constant_ephemeral(self, s_idx):
        end_ind = s_idx.find(" ")
        ind = int(s_idx[1:end_ind])
        if s_idx[0] == "c":
            return self.__all_obj[self.__num_features + ind]
        elif s_idx[0] == "e":
            return self.__all_obj[self.__num_features + self.__num_constants + ind]
        elif s_idx[0] == "x":
            return self.__all_obj[ind]

    def cast(self, s_idx):
        objc = self.get_constant_ephemeral(s_idx)
        start_ind = s_idx.find(" ")
        return objc.cast(s_idx[(start_ind+1):])

    @staticmethod
    def feature_id(s_idx):
        return int(s_idx[1:])

    def sample(self):
        ind = random.randint(0, len(self.__all_obj) - 1)
        return self.__extract(ind)

    def sample_typed(self, provided_type):
        candidates = [i for i in range(len(self.__all_types)) if self.__all_types[i] == provided_type]
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__extract(ind)

    def __extract(self, ind):
        if self.__all_idx[ind][0] == "x" or self.__all_idx[ind][0] == "c":
            return self.__all_idx[ind]
        elif self.__all_idx[ind][0] == "e":
            return self.__all_idx[ind] + str(self.__all_obj[ind]())


class Primitive:
    def __init__(self, name, return_type, parameter_types, function):
        if not(re.search(r'^[xce]\d+', name) is None):
            raise AttributeError(f"Invalid name. {name} is not a valid name for a primitive. Please avoid starting the name with either x or c or e followed by a number.")
        self.__name = name
        self.__arity = len(parameter_types)
        self.__return_type = return_type
        self.__parameter_types = parameter_types
        self.__function = function

    def __str__(self):
        return f"{self.__name} : {self.__parameter_types} --> {self.__return_type}"

    def __call__(self, *args):
        return self.__function(*args)

    def __len__(self):
        return self.__arity

    def arity(self):
        return self.__arity

    def name(self):
        return self.__name

    def return_type(self):
        return self.__return_type

    def parameter_types(self):
        return self.__parameter_types


class PrimitiveSet:
    def __init__(self, primitives, return_type):
        self.__primitive_dict = {p.name(): p for p in primitives}
        self.__primitives = primitives
        self.__num_primitives = len(primitives)
        self.__return_type = return_type

    def __str__(self):
        return f"N. Primitives: {self.__num_primitives} - Return type: {self.__return_type}."

    def __len__(self):
        return self.__num_primitives

    def num_primitives(self):
        return self.__num_primitives

    def return_type(self):
        return self.__return_type

    def get_primitive(self, name):
        return self.__primitive_dict[name]

    def sample(self):
        ind = random.randint(0, len(self.__primitives) - 1)
        return self.__primitives[ind]

    def sample_root(self):
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].return_type() == self.__return_type]
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def sample_typed(self, provided_type):
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].return_type() == provided_type]
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def sample_parameter_typed(self, provided_types):
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].parameter_types() == provided_types]
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def is_primitive(self, s_idx):
        names = [p.name() for p in self.__primitives]
        return s_idx in names


# ==============================================================================================================
# RANDOM TREE GENERATOR
# ==============================================================================================================

def gen_half_half(primitive_set, terminal_set, max_degree, min_height=2, max_height=5):
    ind = random.randint(0, 1)
    if ind == 0:
        return gen_full(primitive_set, terminal_set, max_degree, min_height, max_height)
    else:
        return gen_grow(primitive_set, terminal_set, max_degree, min_height, max_height)


def gen_full(primitive_set, terminal_set, max_degree, min_height=2, max_height=5):
    if not (min_height <= max_height):
        raise AttributeError("Min height must be less than or equal to max height.")
    if min_height <= 1:
        raise AttributeError("Min height must be a positive number of at least 2.")
    if min_height == max_height:
        height = min_height
    else:
        height = random.randint(min_height, max_height)
    tree = [[primitive_set.sample_root().name()]]
    for layer_ind in range(1, height):
        curr_layer = [""]*(max_degree**layer_ind)
        previous_layer = tree[layer_ind - 1]
        parents = [(iii, primitive_set.get_primitive(previous_layer[iii])) for iii in range(len(previous_layer)) if previous_layer[iii] != ""]
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
        tree.append([""]*(max_degree**layer_ind))
    return PrimitiveTree(max_degree, max_height, tree)


def gen_grow(primitive_set, terminal_set, max_degree, min_height=2, max_height=5):
    if not (min_height <= max_height):
        raise AttributeError("Min height must be less than or equal to max height.")
    if min_height <= 1:
        raise AttributeError("Min height must be a positive number of at least 2.")
    if min_height == max_height:
        height = min_height
    else:
        height = random.randint(min_height, max_height)
    tree = [[primitive_set.sample_root().name()]]
    expand = [[True]]
    isMinHeightReached = False
    for layer_ind in range(1, height):
        curr_layer = [""] * (max_degree ** layer_ind)
        curr_expand = [False] * (max_degree ** layer_ind)
        if layer_ind + 1 == min_height:
            isMinHeightReached = True
        previous_layer = tree[layer_ind - 1]
        previous_expand = expand[layer_ind - 1]
        parents = [(iii, previous_expand[iii], primitive_set.get_primitive(previous_layer[iii])) for iii in range(len(previous_layer)) if previous_layer[iii] != "" and primitive_set.is_primitive(previous_layer[iii])]
        if not(isMinHeightReached):
            to_expand_necesserly = random.randint(0, len(parents) - 1)
        else:
            to_expand_necesserly = -1
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
                        if random.random() <= 0.30:
                            curr_layer[start_ind] = terminal_set.sample_typed(t)
                            curr_expand[start_ind] = False
                        else:
                            curr_layer[start_ind] = primitive_set.sample_typed(t).name()
                            if random.random() < 0.50:
                                curr_expand[start_ind] = True
                            else:
                                curr_expand[start_ind] = False
                    elif to_expand_necesserly == p_i and to_expand_necesserly_param == t_i:
                        curr_layer[start_ind] = primitive_set.sample_typed(t).name()
                        curr_expand[start_ind] = True
                    else:
                        curr_layer[start_ind] = terminal_set.sample_typed(t)
                        curr_expand[start_ind] = False
                start_ind += 1
        tree.append(curr_layer)
        expand.append(curr_expand)
    for layer_ind in range(height, max_height):
        tree.append([""]*(max_degree**layer_ind))
    return PrimitiveTree(max_degree, max_height, tree)


# ==============================================================================================================
# TREE
# ==============================================================================================================


class AbstractTree(ABC):
    def __init__(self, max_degree, max_depth):
        self.__max_degree = max_degree
        self.__max_depth = max_depth

        self.__max_number_of_nodes = 0
        for i in range(self.__max_depth):
            if i == self.__max_depth - 1:
                self.__max_breadth = self.__max_degree ** i
            self.__max_number_of_nodes += self.__max_degree ** i

    def print(self):
        print(self.__str__())

    def max_degree(self):
        return self.__max_degree

    def max_depth(self):
        return self.__max_depth

    def max_number_of_nodes(self):
        return self.__max_number_of_nodes

    def max_breadth(self):
        return self.__max_breadth

    @abstractmethod
    def layer(self, layer_ind):
        pass

    @abstractmethod
    def root(self):
        pass

    @abstractmethod
    def number_of_nodes(self):
        pass

    @abstractmethod
    def number_of_nodes_at_layer(self, layer_ind):
        pass

    @abstractmethod
    def depth(self):
        pass

    @abstractmethod
    def actual_max_breadth(self):
        pass

    @abstractmethod
    def actual_max_degree(self):
        pass

    @abstractmethod
    def layer_of_max_breadth(self):
        pass

    @abstractmethod
    def leaf_nodes(self):
        pass

    @abstractmethod
    def internal_nodes(self):
        pass

    @abstractmethod
    def node(self, layer_ind, node_ind):
        pass

    @abstractmethod
    def is_leaf(self, layer_ind, node_ind):
        pass

    @abstractmethod
    def siblings(self, layer_ind, node_ind):
        pass

    @abstractmethod
    def children(self, layer_ind, node_ind):
        pass

    @abstractmethod
    def parent(self, layer_ind, node_ind):
        pass


class PrimitiveTree(AbstractTree):
    def __init__(self, max_degree, max_depth, data):
        super().__init__(max_degree, max_depth)
        # [ ["+"],
        # ["+", "*"],
        # ["-", "x3", "c2 20", "^2"],
        # ["x0", "x5", "", "", "", "", "e0 0.24535563", ""] ]
        self.__tree = data

    def __len__(self):
        return self.number_of_nodes()

    def __str__(self):
        s = ""
        n_indent = self.depth()-1
        s += "\n"
        for i in range(self.depth()):
            for _ in range(n_indent):
                s += "\t"
            n_indent -= 1
            curr_layer = self.layer(i)
            curr_elem = ["["+elem+"]\t" for elem in curr_layer if elem != ""]
            for elem in curr_elem:
                s += elem
            s += "\n"
        return s

    def __check_layer_index_with_max_depth(self, layer_ind):
        if not(0 <= layer_ind < self.max_depth()):
            raise IndexError(f"{layer_ind} is out of range as layer index.")

    def __check_layer_index_with_actual_depth(self, layer_ind):
        if not(0 <= layer_ind < self.depth()):
            raise IndexError(f"{layer_ind} is out of range as layer index.")

    def layer(self, layer_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        return self.__tree[layer_ind]

    def root(self):
        return self.__tree[0][0]

    def number_of_nodes(self):
        n_nodes = 0
        for i in range(self.max_depth()):
            nodes = self.layer(i)
            n_nodes += sum([1 if n != "" else 0 for n in nodes])
        return n_nodes

    def number_of_nodes_at_layer(self, layer_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        return sum([1 if n != "" else 0 for n in self.layer(layer_ind)])

    def depth(self):
        for i in range(self.max_depth()):
            if all([n == "" for n in self.__tree[i]]):
                return i
        return self.max_depth()

    def actual_max_breadth(self):
        max_layer = -1000000
        for i in range(self.max_depth()):
            n_nodes = self.number_of_nodes_at_layer(i)
            if ( n_nodes > max_layer):
                max_layer = n_nodes
        return max_layer

    def actual_max_degree(self):
        max_degree = -1000000
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
            for j in range(len(ind)):
                if len(self.children(i, j)) > max_degree:
                    max_degree = len(self.children(i, j))
        return max_degree

    def layer_of_max_breadth(self):
        max_layer = -1000000
        ind = -1
        for i in range(self.max_depth()):
            n_nodes = self.number_of_nodes_at_layer(i)
            if (n_nodes > max_layer):
                max_layer = n_nodes
                ind = i
        return ind

    def leaf_nodes(self):
        leaf_nodes = []
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
            for j in range(len(ind)):
                if self.is_leaf(i, j):
                    leaf_nodes.append(curr_layer[ind[j]])
        return leaf_nodes

    def internal_nodes(self):
        internal_nodes = []
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
            for j in range(len(ind)):
                if not(self.is_leaf(i, j)):
                    internal_nodes.append(curr_layer[ind[j]])
        return internal_nodes

    def node(self, layer_ind, node_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        elem = [curr_layer[iii] for iii in elem_ind]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        curr_node = elem[node_ind]
        return curr_node

    def is_leaf(self, layer_ind, node_ind):
        return len(self.children(layer_ind, node_ind)) == 0

    def siblings(self, layer_ind, node_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        elem = [curr_layer[iii] for iii in elem_ind]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        curr_node = elem[node_ind]
        return elem[:node_ind], curr_node, elem[(node_ind+1):]

    def children(self, layer_ind, node_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        if layer_ind == self.depth() - 1:
            return []
        curr_layer = self.layer(layer_ind)
        next_layer = self.layer(layer_ind + 1)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem_ind)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        ind = elem_ind[node_ind]
        children = []
        start_ind = self.max_degree() * ind
        for i in range(start_ind, start_ind + self.max_degree()):
            if next_layer[i] != 0:
                children.append(next_layer[i])
        return children

    def parent(self, layer_ind, node_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        if layer_ind == 0:
            return None
        curr_layer = self.layer(layer_ind)
        previous_layer = self.layer(layer_ind - 1)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem_ind)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        ind_of_node = elem_ind[node_ind]
        curr_ind = ind_of_node//self.max_degree()
        return previous_layer[curr_ind]


if __name__ == "__main__":
    constants_0 = [Constant("five", 5.0), Constant("ten", 10.0)]
    ephemeral_0 = [Ephemeral("epm0", lambda: random.random()), Ephemeral("epm1", lambda: float(random.randint(0, 5)))]

    terminal_set_0 = TerminalSet([float]*10, constants_0, ephemeral_0)

    primitives_0 = [Primitive("+", float, [float, float], lambda x, y: x + y),
                  Primitive("-", float, [float, float], lambda x, y: x - y),
                  Primitive("*", float, [float, float], lambda x, y: x * y),
                  Primitive("^2", float, [float], lambda x: x ** 2),
                  Primitive("*2", float, [float], lambda x: x * 2.0),
                  Primitive("/2", float, [float], lambda x: x / 2.0),
                  Primitive("*3", float, [float], lambda x: x * 3.0),
                  Primitive("/3", float, [float], lambda x: x / 3.0)
                  ]

    primitive_set_0 = PrimitiveSet(primitives_0, float)

    gen_half_half(primitive_set_0, terminal_set_0, 2, 2, 5).print()

