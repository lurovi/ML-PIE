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
    def __get_right_key_from_terminal(s: str) -> str:
        is_primitive = Primitive.check_valid_primitive_name(s)
        if is_primitive:
            return s
        else:
            if s[0] == "x":
                return s
            elif s[0] == "c":
                return s[:s.find(" ")]
            elif s[0] == "e":
                return s[:s.find(" ")]
    arr = []
    for layer_ind in range(tree.max_depth()):
        curr_layer = tree.layer(layer_ind)
        curr_layer = [0.0 if n == "" else weights[__get_right_key_from_terminal(n)] for n in curr_layer]
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
    for layer_ind in range(tree.max_depth()):
        curr_layer = tree.layer(layer_ind)
        curr_weights = weights[layer_ind]
        curr_layer = [0.0 if n == "" else curr_weights[__get_right_key_from_terminal(n)] for n in curr_layer]
        arr.extend(curr_layer)
    return torch.tensor(arr, dtype=torch.float32)


def total_level_wise_weights_tree_converter(tree: PrimitiveTree, weights: List[Dict[str, float]]) -> torch.Tensor:
    # just like total_level_weights_tree_converter but now there are different weights for each layer
    # that is, the same type of node may have different weight depending on the layer in which it is collocated
    # now weights is a list of dict, one per layer, starting from the root (first element of the list) to the bottom
    def __get_right_key_from_terminal(s: str) -> str:
        is_primitive = Primitive.check_valid_primitive_name(s)
        if is_primitive:
            return s
        else:
            if s[0] == "x":
                return s
            elif s[0] == "c":
                return s[:s.find(" ")]
            elif s[0] == "e":
                return s[:s.find(" ")]
    arr = []
    for layer_ind in range(tree.max_depth()):
        curr_layer = tree.layer(layer_ind)
        curr_weights = weights[layer_ind]
        curr_layer = [0.0 if n == "" else curr_weights[__get_right_key_from_terminal(n)] for n in curr_layer]
        arr.extend(curr_layer)
    return torch.tensor(arr, dtype=torch.float32)


def one_hot_tree(tree: PrimitiveTree):
    def __get_right_key_from_terminal(s: str) -> str:
        is_primitive = Primitive.check_valid_primitive_name(s)
        if is_primitive:
            return s
        else:
            if s[0] == "x":
                return s
            elif s[0] == "c":
                return s[:s.find(" ")]
            elif s[0] == "e":
                return s[:s.find(" ")]
    num_primitives = tree.primitive_set().num_primitives()
    num_features = tree.terminal_set().num_features()
    num_constants = tree.terminal_set().num_constants()
    num_ephemeral = tree.terminal_set().num_ephemeral()
    tot_attr = num_primitives + num_features + num_constants + num_ephemeral
    tot_attr_names = tree.primitive_set().primitive_names() + list(range(tree.terminal_set().num_features())) + list(range(tree.terminal_set().num_constants())) + list(range(tree.terminal_set().num_ephemeral()))
    dic = {"": [0.0]*tot_attr}
    t = 0
    for p in tot_attr_names:
        if t < num_primitives:
            curr_key = p
        elif num_primitives <= t < num_primitives + num_features:
            curr_key = "x" + str(p)
        elif num_primitives + num_features <= t < num_primitives + num_features + num_constants:
            curr_key = "c" + str(p)
        elif num_primitives + num_features + num_constants <= t < num_primitives + num_features + num_constants + num_ephemeral:
            curr_key = "e" + str(p)
        li = [0.0] * tot_attr
        li[t] = 1.0
        dic[curr_key] = li
        t += 1
    arr = []
    for layer_ind in range(tree.max_depth()):
        curr_layer = tree.layer(layer_ind)
        curr_arr = []
        for n in curr_layer:
            curr_arr.extend(dic[__get_right_key_from_terminal(n)])
        arr.extend(curr_arr)
    return torch.tensor(arr, dtype=torch.float32)


def one_hot_tree_as_image(tree: PrimitiveTree):
    def __get_right_key_from_terminal(s: str) -> str:
        is_primitive = Primitive.check_valid_primitive_name(s)
        if is_primitive:
            return s
        else:
            if s[0] == "x":
                return s
            elif s[0] == "c":
                return s[:s.find(" ")]
            elif s[0] == "e":
                return s[:s.find(" ")]
    num_primitives = tree.primitive_set().num_primitives()
    num_features = tree.terminal_set().num_features()
    num_constants = tree.terminal_set().num_constants()
    num_ephemeral = tree.terminal_set().num_ephemeral()
    tot_attr = num_primitives + num_features + num_constants + num_ephemeral
    tot_attr_names = tree.primitive_set().primitive_names() + list(range(tree.terminal_set().num_features())) + list(range(tree.terminal_set().num_constants())) + list(range(tree.terminal_set().num_ephemeral()))
    dic = {"": [0.0]*tot_attr}
    t = 0
    for p in tot_attr_names:
        if t < num_primitives:
            curr_key = p
        elif num_primitives <= t < num_primitives + num_features:
            curr_key = "x" + str(p)
        elif num_primitives + num_features <= t < num_primitives + num_features + num_constants:
            curr_key = "c" + str(p)
        elif num_primitives + num_features + num_constants <= t < num_primitives + num_features + num_constants + num_ephemeral:
            curr_key = "e" + str(p)
        li = [0.0]*tot_attr
        li[t] = 1.0
        dic[curr_key] = li
        t += 1
    max_breadth = tree.max_breadth()
    height = tree.max_depth()
    width = max_breadth*tot_attr
    arr = []
    for layer_ind in range(height):
        curr_layer = tree.layer(layer_ind)
        curr_arr = []
        for i in range(len(curr_layer)):
            curr_arr.extend(dic[__get_right_key_from_terminal(curr_layer[i])])
        for i in range(len(curr_layer)*tot_attr, width):
            curr_arr.append(0.0)
        arr.append(curr_arr)
    return torch.tensor([arr], dtype=torch.float32)


# ==============================================================================================================
# LABELS CALCULATOR
# ==============================================================================================================

# weights is a dictionary or list of dictionary while data is a list of primitive trees (this holds for methods of this section)
def transform_with_weights(data, weights):
    tre = []
    for t in data:
        c = total_weights_tree_converter(t, weights)
        s = c.sum().item()
        tre.append((c, s))
    return tre


def transform_with_weights_level_wise(data, weights):
    tre = []
    for t in data:
        c = total_level_wise_weights_tree_converter(t, weights)
        s = c.sum().item()
        tre.append((c, s))
    return tre


def compute_labels_from_one_hot(data, weights):
    tre = []
    for t in data:
        c = total_weights_tree_converter(t, weights)
        s = c.sum().item()
        tre.append((one_hot_tree(t), s))
    return tre


def compute_labels_from_one_hot_level_wise(data, weights):
    tre = []
    for t in data:
        c = total_level_wise_weights_tree_converter(t, weights)
        s = c.sum().item()
        tre.append((one_hot_tree(t), s))
    return tre


if __name__ == "__main__":
    constants_0 = [Constant("five", 5.0), Constant("ten", 10.0)]
    ephemeral_0 = [Ephemeral("epm0", lambda: random.random()), Ephemeral("epm1", lambda: float(random.randint(0, 4)))]

    terminal_set_0 = TerminalSet([float] * 7, constants_0, ephemeral_0)

    primitives_0 = [Primitive("+", float, [float, float], lambda x, y: x + y),
                    Primitive("-", float, [float, float], lambda x, y: x - y),
                    Primitive("*", float, [float, float], lambda x, y: x * y),
                    Primitive("max", float, [float, float], lambda x, y: max(x, y)),
                    Primitive("min", float, [float, float], lambda x, y: min(x, y)),
                    Primitive("abs", float, [float], lambda x: abs(x)),
                    Primitive("neg", float, [float], lambda x: -x),
                    Primitive("^2", float, [float], lambda x: x ** 2),
                    Primitive("*2", float, [float], lambda x: x * 2.0),
                    Primitive("/2", float, [float], lambda x: x / 2.0),
                    ]

    primitive_set_0 = PrimitiveSet(primitives_0, float)

    tr = gen_half_half(primitive_set_0, terminal_set_0, 3, 8)

    print(tr)

    d_1 = {"+": 0.90, "-": 0.70, "*": 0.60, "max": 0.20, "min": 0.20, "abs": 0.15, "neg": 0.65,
           "^2": 0.18, "*2": 0.65, "/2": 0.57,
           "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20}

    d_2 = {"+": 0.90, "-": 0.70, "*": 0.60, "max": 0.20, "min": 0.20, "abs": 0.15, "neg": 0.65,
           "^2": 0.18, "*2": 0.65, "/2": 0.57,
           "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
           "c0": 0.23, "c1": 0.23,
           "e0": 0.12, "e1": 0.12}

    d_3 = [{"+": 0.90, "-": 0.70, "*": 0.55, "max": 0.21, "min": 0.21, "abs": 0.15, "neg": 0.68,
           "^2": 0.16, "*2": 0.65, "/2": 0.57,
           "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20},
           {"+": 0.88, "-": 0.68, "*": 0.55, "max": 0.23, "min": 0.23, "abs": 0.15, "neg": 0.67,
            "^2": 0.18, "*2": 0.65, "/2": 0.57,
            "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20},
           {"+": 0.86, "-": 0.66, "*": 0.56, "max": 0.24, "min": 0.24, "abs": 0.15, "neg": 0.65,
            "^2": 0.18, "*2": 0.63, "/2": 0.56,
            "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20},
           {"+": 0.85, "-": 0.65, "*": 0.56, "max": 0.25, "min": 0.25, "abs": 0.16, "neg": 0.65,
            "^2": 0.18, "*2": 0.63, "/2": 0.55,
            "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20},
           {"+": 0.84, "-": 0.64, "*": 0.57, "max": 0.26, "min": 0.26, "abs": 0.17, "neg": 0.64,
            "^2": 0.19, "*2": 0.62, "/2": 0.54,
            "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20},
           {"+": 0.82, "-": 0.62, "*": 0.57, "max": 0.27, "min": 0.27, "abs": 0.18, "neg": 0.64,
            "^2": 0.19, "*2": 0.62, "/2": 0.54,
            "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20},
           {"+": 0.81, "-": 0.61, "*": 0.58, "max": 0.28, "min": 0.28, "abs": 0.20, "neg": 0.63,
            "^2": 0.20, "*2": 0.61, "/2": 0.53,
            "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20},
           {"+": 0.80, "-": 0.60, "*": 0.60, "max": 0.30, "min": 0.30, "abs": 0.20, "neg": 0.63,
            "^2": 0.22, "*2": 0.59, "/2": 0.52,
            "FEATURE": 0.80, "CONSTANT": 0.30, "EPHEMERAL": 0.20}
           ]

    d_4 = [{"+": 0.90, "-": 0.70, "*": 0.55, "max": 0.21, "min": 0.21, "abs": 0.15, "neg": 0.68,
           "^2": 0.16, "*2": 0.65, "/2": 0.57,
           "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
           "c0": 0.23, "c1": 0.23,
           "e0": 0.12, "e1": 0.12},
           {"+": 0.88, "-": 0.68, "*": 0.55, "max": 0.23, "min": 0.23, "abs": 0.15, "neg": 0.67,
            "^2": 0.18, "*2": 0.65, "/2": 0.57,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
           "c0": 0.23, "c1": 0.23,
           "e0": 0.12, "e1": 0.12},
           {"+": 0.86, "-": 0.66, "*": 0.56, "max": 0.24, "min": 0.24, "abs": 0.15, "neg": 0.65,
            "^2": 0.18, "*2": 0.63, "/2": 0.56,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
           "c0": 0.23, "c1": 0.23,
           "e0": 0.12, "e1": 0.12},
           {"+": 0.85, "-": 0.65, "*": 0.56, "max": 0.25, "min": 0.25, "abs": 0.16, "neg": 0.65,
            "^2": 0.18, "*2": 0.63, "/2": 0.55,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
           "c0": 0.23, "c1": 0.23,
           "e0": 0.12, "e1": 0.12},
           {"+": 0.84, "-": 0.64, "*": 0.57, "max": 0.26, "min": 0.26, "abs": 0.17, "neg": 0.64,
            "^2": 0.19, "*2": 0.62, "/2": 0.54,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
            "c0": 0.23, "c1": 0.23,
            "e0": 0.12, "e1": 0.12},
           {"+": 0.82, "-": 0.62, "*": 0.57, "max": 0.27, "min": 0.27, "abs": 0.18, "neg": 0.64,
            "^2": 0.19, "*2": 0.62, "/2": 0.54,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
           "c0": 0.23, "c1": 0.23,
           "e0": 0.12, "e1": 0.12},
           {"+": 0.81, "-": 0.61, "*": 0.58, "max": 0.28, "min": 0.28, "abs": 0.20, "neg": 0.63,
            "^2": 0.20, "*2": 0.61, "/2": 0.53,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
           "c0": 0.23, "c1": 0.23,
           "e0": 0.12, "e1": 0.12},
           {"+": 0.80, "-": 0.60, "*": 0.60, "max": 0.30, "min": 0.30, "abs": 0.20, "neg": 0.63,
            "^2": 0.22, "*2": 0.59, "/2": 0.52,
            "x0": 0.80, "x1": 0.80, "x2": 0.80, "x3": 0.80, "x4": 0.80, "x5": 0.80, "x6": 0.80,
           "c0": 0.23, "c1": 0.23,
           "e0": 0.12, "e1": 0.12}
           ]

    print(one_hot_tree_as_image(tr))




