import torch
from typing import Callable, Any, Iterable, Dict, List
from gputil.tree import *


# ==============================================================================================================
# TREE CONVERTER
# ==============================================================================================================

def simple_weights_tree_converter(tree: PrimitiveTree, weights: Dict[str, float]) -> torch.Tensor:
    # weights is a dict where the key is a string representing the name of a primitive and the
    # value is a float that represents a weight for that particular primitive
    # moreover, weights must contain two special <key,value> pairs, one with key equal to "CONSTANT"
    # that contains the weight of a constant, and one with key equal to "EPHEMERAL"
    # that contains the weight for an ephemeral.
    # moreover, weights must contain a <key,value> pair with key "FEATURE" and a weight to assign to
    # appearance of a generic feature

    # this method generates a 1-D torch tensor with length equal to the max number of nodes of the tree
    # this tensor is flattened in breadth first order, all layers are concatenated from top to bottom
    # and from left to right, including blank nodes (represented with 0)

    # each node is replaced with the corresponding weight
    def __get_right_key_from_terminal(s: str) -> str:
        is_primitive = Primitive.check_valid_primitive_name(s)
        if is_primitive:
            return s
        else:
            if s[0] == "x":
                return "FEATURE"
            elif s[0] == "c":
                return "CONSTANT"
            elif s[0] == "e":
                return "EPHEMERAL"
    arr = []
    for layer_ind in range(tree.max_depth()):
        curr_layer = tree.layer(layer_ind)
        curr_layer = [0.0 if n == "" else weights[__get_right_key_from_terminal(n)] for n in curr_layer]
        arr.extend(curr_layer)
    return torch.tensor(arr, dtype=torch.float32)


def total_weights_tree_converter(tree: PrimitiveTree, weights: Dict[str, float]) -> torch.Tensor:
    # the same as before but different weights are applied to different terminals too
    # therefore, in the dictionary there must be keys for all features, constants and ephemeral
    arr = []
    for layer_ind in range(tree.max_depth()):
        curr_layer = tree.layer(layer_ind)
        curr_layer = [0.0 if n == "" else weights[n] for n in curr_layer]
        arr.extend(curr_layer)
    return torch.tensor(arr, dtype=torch.float32)


def simple_level_wise_weights_tree_converter(tree: PrimitiveTree, weights: List[Dict[str, float]]) -> torch.Tensor:
    # just like simple_level_weights_tree_converter but now there are different weights for each layer
    # that is, the same type of node may have different weight depending on the layer in which it is collocated
    # now weights is a list of dict, one per layer, starting from the root (first element of the list) to the bottom
    def __get_right_key_from_terminal(s: str) -> str:
        is_primitive = Primitive.check_valid_primitive_name(s)
        if is_primitive:
            return s
        else:
            if s[0] == "x":
                return "FEATURE"
            elif s[0] == "c":
                return "CONSTANT"
            elif s[0] == "e":
                return "EPHEMERAL"
    arr = []
    for layer_ind in range(tree.depth()):
        curr_layer = tree.layer(layer_ind)
        curr_weights = weights[layer_ind]
        curr_layer = [0.0 if n == "" else curr_weights[__get_right_key_from_terminal(n)] for n in curr_layer]
        arr.extend(curr_layer)
    for layer_ind in range(tree.depth(), tree.max_depth()):
        arr.extend([0.0]*len(tree.layer(layer_ind)))
    return torch.tensor(arr, dtype=torch.float32)


def total_level_wise_weights_tree_converter(tree: PrimitiveTree, weights: List[Dict[str, float]]) -> torch.Tensor:
    # just like total_level_weights_tree_converter but now there are different weights for each layer
    # that is, the same type of node may have different weight depending on the layer in which it is collocated
    # now weights is a list of dict, one per layer, starting from the root (first element of the list) to the bottom
    arr = []
    for layer_ind in range(tree.max_depth()):
        curr_layer = tree.layer(layer_ind)
        curr_weights = weights[layer_ind]
        curr_layer = [0.0 if n == "" else curr_weights[n] for n in curr_layer]
        arr.extend(curr_layer)
    for layer_ind in range(tree.depth(), tree.max_depth()):
        arr.extend([0.0]*len(tree.layer(layer_ind)))
    return torch.tensor(arr, dtype=torch.float32)

def one_hot_tree(tree: PrimitiveTree):
    num_primitives = tree.primitive_set().num_primitives()
    num_features = tree.terminal_set().num_features()
    num_constants = tree.terminal_set().num_constants()
    num_ephemeral = tree.terminal_set().num_ephemeral()
    tot_attr = num_primitives + num_features + num_constants + num_ephemeral
    dic = {"": [0.0]*tot_attr}
    for p in tree.primitive_set().primitive_names():
        pass


if __name__ == "__main__":
    constants_0 = [Constant("five", 5.0), Constant("ten", 10.0)]
    ephemeral_0 = [Ephemeral("epm0", lambda: random.random()), Ephemeral("epm1", lambda: float(random.randint(0, 5)))]

    terminal_set_0 = TerminalSet([float] * 10, constants_0, ephemeral_0)

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

    tr = gen_half_half(primitive_set_0, terminal_set_0, 2, 2, 5)

    print(tr)
    d = {"+": 0.90, "-": 0.70, "*": 0.60, "^2": 0.20, "*2": 0.50, "/2": 0.50, "*3": 0.50, "/3": 0.50,
         "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20}
    print(simple_weights_tree_converter(tr, d))
