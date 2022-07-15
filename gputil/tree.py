import random
import re
from abc import ABC, abstractmethod
from typing import Callable, Any, List


# ==============================================================================================================
# BUILDING BLOCKS
# ==============================================================================================================


class Constant:
    def __init__(self, name: str, val: Any):
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

    def cast(self, val: Any):
        return self.__type(val)


class Ephemeral:
    def __init__(self, name: str, func: Callable):
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

    def cast(self, val: Any):
        return self.__type(val)


class TerminalSet:
    def __init__(self, feature_types: List[Any], constants: List[Constant], ephemeral: List[Ephemeral]):
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

    def get_constant_ephemeral(self, s_idx: str) -> Any:
        end_ind = s_idx.find(" ")
        ind = int(s_idx[1:end_ind])
        if s_idx[0] == "c":
            return self.__all_obj[self.__num_features + ind]
        elif s_idx[0] == "e":
            return self.__all_obj[self.__num_features + self.__num_constants + ind]
        elif s_idx[0] == "x":
            return self.__all_obj[ind]

    def cast(self, s_idx: str) -> Any:
        objc = self.get_constant_ephemeral(s_idx)
        start_ind = s_idx.find(" ")
        return objc.cast(s_idx[(start_ind+1):])

    @staticmethod
    def feature_id(s_idx: str) -> int:
        return int(s_idx[1:])

    def sample(self) -> str:
        ind = random.randint(0, len(self.__all_obj) - 1)
        return self.__extract(ind)

    def sample_typed(self, provided_type: Any) -> str:
        candidates = [i for i in range(len(self.__all_types)) if self.__all_types[i] == provided_type]
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__extract(ind)

    def __extract(self, ind: int) -> str:
        if self.__all_idx[ind][0] == "x" or self.__all_idx[ind][0] == "c":
            return self.__all_idx[ind]
        elif self.__all_idx[ind][0] == "e":
            return self.__all_idx[ind] + str(self.__all_obj[ind]())


class Primitive:
    def __init__(self, name: str, return_type: Any, parameter_types: List[Any], function: Callable):
        if not(Primitive.check_valid_primitive_name(name)):
            raise AttributeError(f"Invalid name. {name} is not a valid name for a primitive. Please avoid starting the name with either x or c or e followed by a number.")
        self.__name = name
        self.__arity = len(parameter_types)
        self.__return_type = return_type
        self.__parameter_types = parameter_types
        self.__function = function

    @staticmethod
    def check_valid_primitive_name(s: str) -> bool:
        return re.search(r'^[xce]\d+', s) is None

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
    def __init__(self, primitives: List[Primitive], return_type: Any):
        self.__primitive_dict = {p.name(): p for p in primitives}
        self.__primitive_idx = {primitives[i].name(): i for i in range(len(primitives))}
        self.__primitives = primitives
        self.__primitive_names = [p.name() for p in primitives]
        self.__num_primitives = len(primitives)
        self.__return_type = return_type
        self.__max_arity = max([p.arity() for p in primitives])

    def __str__(self):
        return f"N. Primitives: {self.__num_primitives} - Return type: {self.__return_type}."

    def __len__(self):
        return self.__num_primitives

    def max_arity(self):
        return self.__max_arity

    def primitive_names(self):
        return self.__primitive_names

    def num_primitives(self):
        return self.__num_primitives

    def return_type(self):
        return self.__return_type

    def get_primitive(self, name: str) -> Primitive:
        return self.__primitive_dict[name]

    def get_primitive_idx(self, name: str) -> Primitive:
        return self.__primitive_idx[name]

    def sample(self) -> Primitive:
        ind = random.randint(0, len(self.__primitives) - 1)
        return self.__primitives[ind]

    def sample_root(self) -> Primitive:
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].return_type() == self.__return_type]
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def sample_typed(self, provided_type: Any) -> Primitive:
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].return_type() == provided_type]
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def sample_parameter_typed(self, provided_types: Any) -> Primitive:
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].parameter_types() == provided_types]
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def is_primitive(self, s_idx: str) -> bool:
        names = [p.name() for p in self.__primitives]
        return s_idx in names

# ==============================================================================================================
# TREE
# ==============================================================================================================


'''
class AbstractTree(ABC):      

    @abstractmethod
    def print_as_tree(self):
        pass

    @abstractmethod
    def layer(self, layer_ind: int):
        pass

    @abstractmethod
    def root(self):
        pass

    @abstractmethod
    def flatten(self):
        pass

    @abstractmethod
    def number_of_nodes(self):
        pass

    @abstractmethod
    def number_of_nodes_at_layer(self, layer_ind: int):
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
    def node(self, layer_ind: int, node_ind: int):
        pass

    @abstractmethod
    def is_leaf(self, layer_ind: int, node_ind: int):
        pass

    @abstractmethod
    def siblings(self, layer_ind: int, node_ind: int):
        pass

    @abstractmethod
    def children(self, layer_ind: int, node_ind: int):
        pass

    @abstractmethod
    def parent(self, layer_ind: int, node_ind: int):
        pass
'''


class PrimitiveTree:
    def __init__(self, data: List[List[str]], primitive_set: PrimitiveSet, terminal_set: TerminalSet):
        self.__max_degree = primitive_set.max_arity()
        self.__max_depth = len(data)

        self.__max_number_of_nodes = 0
        for i in range(self.__max_depth):
            if i == self.__max_depth - 1:
                self.__max_breadth = self.__max_degree ** i
            self.__max_number_of_nodes += self.__max_degree ** i

        #    [ ["+"],
        #   ["+", "*"],
        #  ["-", "x3", "c2 20", "^2"],
        # ["x0", "x5", "", "", "", "", "e0 0.24535563", ""] ]
        self.__tree = data
        self.__primitive_set = primitive_set
        self.__terminal_set = terminal_set

    def max_degree(self):
        return self.__max_degree

    def max_depth(self):
        return self.__max_depth

    def max_number_of_nodes(self):
        return self.__max_number_of_nodes

    def max_breadth(self):
        return self.__max_breadth

    def primitive_set(self):
        return self.__primitive_set

    def terminal_set(self):
        return self.__terminal_set

    def __len__(self):
        return self.number_of_nodes()

    def print_as_text(self):
        return self.__str__()

    def __str__(self):
        s = "  "
        stack = [(0, 0, self.layer(0)[0])]
        while len(stack) > 0:
            val = stack.pop()
            if type(val) == str:
                s += val + "  "
            else:
                curr_i, curr_j, curr_val = val
                children = self.children(curr_i, curr_j)
                s += curr_val + "  "
                if len(children) != 0:
                    stack.append(")")
                    for iii in reversed(range(len(children))):
                        stack.append(children[iii])
                    stack.append("(")
        return s

    def print_as_tree(self):
        s = ""
        n_indent = self.depth()-1
        s += "\n"
        for i in range(self.depth()):
            for _ in range(n_indent):
                s += " "
            n_indent -= 1
            curr_layer = self.layer(i)
            curr_elem = ["["+elem+"]\t" for elem in curr_layer if elem != ""]
            for elem in curr_elem:
                s += elem
            s += "\n"
        return s

    def __check_layer_index_with_max_depth(self, layer_ind: int):
        if not(0 <= layer_ind < self.max_depth()):
            raise IndexError(f"{layer_ind} is out of range as layer index.")

    def __check_layer_index_with_actual_depth(self, layer_ind: int):
        if not(0 <= layer_ind < self.depth()):
            raise IndexError(f"{layer_ind} is out of range as layer index.")

    def layer(self, layer_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        return self.__tree[layer_ind]

    def root(self):
        return self.__tree[0][0]

    def flatten(self):
        tre = []
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            tre.append([n for n in curr_layer if n != ""])
        return tre

    def number_of_nodes(self):
        n_nodes = 0
        for i in range(self.max_depth()):
            nodes = self.layer(i)
            n_nodes += sum([1 if n != "" else 0 for n in nodes])
        return n_nodes

    def count_primitives(self):
        dic = {}
        for p in self.primitive_set().primitive_names():
            dic[p] = 0.0
            for p0 in self.primitive_set().primitive_names():
                lp = sorted([p, p0])
                dic[(lp[0], lp[1])] = 0.0

        for layer_ind in range(self.depth()):
            curr_layer = self.layer(layer_ind)
            elem = [curr_layer[i] for i in range(len(curr_layer)) if curr_layer[i] != ""]
            prim = []
            for i in range(len(elem)):
                if self.primitive_set().is_primitive(elem[i]):
                    prim.append((elem[i], [child[2] for child in self.children(layer_ind, i) if self.primitive_set().is_primitive(child[2])]))
            for pr, child in prim:
                dic[pr] += 1.0
                for c in child:
                    lp = sorted([pr, c])
                    dic[(lp[0], lp[1])] += 1.0
        return dic

    def number_of_nodes_at_layer(self, layer_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
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

    def node(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        elem = [curr_layer[iii] for iii in elem_ind]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        curr_node = elem[node_ind]
        return curr_node

    def is_leaf(self, layer_ind: int, node_ind: int):
        return len(self.children(layer_ind, node_ind)) == 0

    def siblings(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        elem = [curr_layer[iii] for iii in elem_ind]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        curr_node = elem[node_ind]
        return elem[:node_ind], curr_node, elem[(node_ind+1):]

    def children(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        if layer_ind == self.depth() - 1:
            return []
        curr_layer = self.layer(layer_ind)
        next_layer = self.layer(layer_ind + 1)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        next_elem_ind = [iii for iii in range(len(next_layer)) if next_layer[iii] != ""]
        if not (0 <= node_ind < len(elem_ind)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        ind = elem_ind[node_ind]
        children = []
        start_ind = self.max_degree() * ind
        relative_ind = -1
        for i in range(start_ind, start_ind + self.max_degree()):
            if next_layer[i] != "":
                for iii in range(len(next_elem_ind)):
                    if next_elem_ind[iii] == i:
                        relative_ind = iii
                        break
                children.append((layer_ind + 1, relative_ind, next_layer[i]))
        return children

    def parent(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        if layer_ind == 0:
            return None
        curr_layer = self.layer(layer_ind)
        previous_layer = self.layer(layer_ind - 1)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        previous_elem_ind = [iii for iii in range(len(previous_layer)) if previous_layer[iii] != ""]
        if not (0 <= node_ind < len(elem_ind)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        ind_of_node = elem_ind[node_ind]
        curr_ind = ind_of_node//self.max_degree()
        relative_ind = -1
        for iii in range(len(previous_elem_ind)):
            if previous_elem_ind[iii] == curr_ind:
                relative_ind = iii
                break
        return (layer_ind - 1, relative_ind, previous_layer[curr_ind])

    def extract_subtree(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        tre = [[curr_layer[elem[node_ind]]]]
        if self.is_leaf(layer_ind, node_ind):
            return PrimitiveTree(tre, self.primitive_set(), self.terminal_set())
        first_previous_node_abs_index = elem[node_ind]
        curr_layer_ind = layer_ind + 1
        for i in range(self.max_depth() - layer_ind - 1):
            curr_dim = self.max_degree()**(curr_layer_ind - layer_ind)
            start_ind = first_previous_node_abs_index * self.max_degree()
            tre.append(self.layer(curr_layer_ind)[start_ind:start_ind + curr_dim])
            curr_layer_ind += 1
            first_previous_node_abs_index = start_ind
        return PrimitiveTree(tre, self.primitive_set(), self.terminal_set())

    def remove_subtree(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        tre = [[self.__tree[i][j] for j in range(len(self.__tree[i]))] for i in range(len(self.__tree))]
        curr_layer = self.layer(layer_ind)
        elem = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        first_previous_node_abs_index = elem[node_ind]
        tre[layer_ind][elem[node_ind]] = ""
        curr_layer_ind = layer_ind + 1
        for i in range(self.max_depth() - layer_ind - 1):
            curr_dim = self.max_degree() ** (curr_layer_ind - layer_ind)
            start_ind = first_previous_node_abs_index * self.max_degree()
            tre[curr_layer_ind] = tre[curr_layer_ind][:start_ind] + [""]*curr_dim + tre[curr_layer_ind][start_ind+curr_dim:]
            curr_layer_ind += 1
            first_previous_node_abs_index = start_ind
        return PrimitiveTree(tre, self.primitive_set(), self.terminal_set())

    def insert_subtree(self, new_tree: PrimitiveTree, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        tre = [[self.__tree[i][j] for j in range(len(self.__tree[i]))] for i in range(len(self.__tree))]
        curr_layer = self.layer(layer_ind)
        elem = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        first_previous_node_abs_index = elem[node_ind]
        tre[layer_ind][elem[node_ind]] = new_tree[0][0]
        curr_layer_ind = layer_ind + 1
        for i in range(self.max_depth() - layer_ind - 1):
            curr_dim = self.max_degree() ** (curr_layer_ind - layer_ind)
            start_ind = first_previous_node_abs_index * self.max_degree()
            tre[curr_layer_ind] = tre[curr_layer_ind][:start_ind] + new_tree[curr_layer_ind - layer_ind] + tre[curr_layer_ind][start_ind + curr_dim:]
            curr_layer_ind += 1
            first_previous_node_abs_index = start_ind
        return PrimitiveTree(tre, self.primitive_set(), self.terminal_set())

    def extract_counting_features_from_tree(self):
        counting_dic = self.count_primitives()
        number_of_nodes = float(self.number_of_nodes())
        depth = float(self.depth())
        max_breadth = float(self.actual_max_breadth())
        max_degree = float(self.actual_max_degree())
        number_of_leaf_nodes = float(len(self.leaf_nodes()))
        number_of_internal_nodes = float(len(self.internal_nodes()))
        leaf_internal_nodes_ratio = number_of_leaf_nodes / number_of_internal_nodes
        leaf_nodes_perc = number_of_leaf_nodes / number_of_nodes
        degree_breadth_ratio = max_degree / max_breadth
        depth_number_of_nodes_ratio = depth / number_of_nodes
        keys = list(counting_dic.keys())
        single_primitives = []
        couples_primitives = []
        for k in keys:
            if isinstance(k, str):
                single_primitives.append(k)
            elif isinstance(k, tuple):
                couples_primitives.append(k)
            else:
                raise ValueError(f"Invalid key {k} found in counting dictionary key set.")
        single_primitives = sorted(single_primitives)
        couples_primitives = sorted(couples_primitives)
        counts = []
        for p in single_primitives:  # + couples_primitives:
            counts.append(float(counting_dic[p]))
        return counts + [number_of_nodes, depth, max_degree, max_breadth, depth_number_of_nodes_ratio, leaf_internal_nodes_ratio]

    @staticmethod
    def extract_counting_features_from_list_of_trees(trees: List):
        lt = []
        for tree in trees:
            lt.append(tree.extract_counting_features_from_tree())
        return lt

    def compile(self, x: List):
        tre = [[self.__tree[i][j] for j in range(len(self.__tree[i]))] for i in range(len(self.__tree))]
        for layer_ind in reversed(range(self.depth()-1)):
            curr_layer = tre[layer_ind]
            next_layer = tre[layer_ind + 1]
            elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != "" and self.__primitive_set.is_primitive(curr_layer[iii])]
            for i in range(len(elem_ind)):
                curr_ind = elem_ind[i]
                children = []
                start_ind = self.max_degree() * curr_ind
                end_ind = start_ind + self.max_degree()
                for j in range(start_ind, end_ind):
                    child = next_layer[j]
                    if next_layer[j] != "":
                        if isinstance(child, str) and not(Primitive.check_valid_primitive_name(child)):
                            child = child.strip()
                            if not(re.search(r'^x\d+', child) is None):
                                children.append(x[int(child[1:])])
                            elif not(re.search(r'^[ce]\d+\s', child) is None):
                                children.append(self.__terminal_set.cast(child))
                            else:
                                children.append(child)
                        elif isinstance(child, str):
                            children.append(child.strip())
                        else:
                            children.append(child)
                tre[layer_ind][curr_ind] = self.__primitive_set.get_primitive(tre[layer_ind][curr_ind])(*children)
        return tre[0][0]


# ==============================================================================================================
# RANDOM TREE GENERATOR
# ==============================================================================================================


def gen_half_half(primitive_set: PrimitiveSet, terminal_set: TerminalSet, min_height: int, max_height: int) -> PrimitiveTree:
    ind = random.randint(0, 1)
    if ind == 0:
        return gen_full(primitive_set, terminal_set, min_height, max_height)
    else:
        return gen_grow(primitive_set, terminal_set, min_height, max_height)


def gen_full(primitive_set: PrimitiveSet, terminal_set: TerminalSet, min_height: int, max_height: int) -> PrimitiveTree:
    max_degree = primitive_set.max_arity()
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
    return PrimitiveTree(tree, primitive_set, terminal_set)


def gen_grow(primitive_set: PrimitiveSet, terminal_set: TerminalSet, min_height: int, max_height: int) -> PrimitiveTree:
    max_degree = primitive_set.max_arity()
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
                        if random.random() <= 0.20:
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
    return PrimitiveTree(tree, primitive_set, terminal_set)
