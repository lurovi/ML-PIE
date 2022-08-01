import numpy as np
import torch
from typing import Callable, Any, Iterable, Dict, List
from gp.tree import *


# ==============================================================================================================
# TREE CONVERTER
# ==============================================================================================================

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


def total_level_wise_weights_tree_converter(tree: PrimitiveTree, weights: List[Dict[str, float]]) -> torch.Tensor:
    # just like total_level_weights_tree_converter but now there are different weights for each layer
    # that is, the same type of node may have different weight depending on the layer in which it is collocated
    # now weights is a list of dict, one per layer, starting from the root (first element of the list) to the bottom
    def __get_right_key_from_terminal(s: str) -> str:
        if s.strip() == "":
            return ""
        is_primitive = Primitive.check_valid_primitive_name(s)
        if is_primitive:
            return s
        else:
            return "x0"
            #if s[0] == "x":
            #    return s
            #elif s[0] == "c":
            #    return s[:s.find(" ")]
            #elif s[0] == "e":
            #    return s[:s.find(" ")]
    arr = []
    for layer_ind in range(tree.max_depth()):
        curr_layer = tree.layer(layer_ind)
        curr_weights = weights[layer_ind]
        curr_layer = [0.0 if n == "" else curr_weights[__get_right_key_from_terminal(n)] for n in curr_layer]
        arr.extend(curr_layer)
    return np.array(arr, dtype=np.float32)


def one_hot_tree(tree: PrimitiveTree):
    def __get_right_key_from_terminal(s: str) -> str:
        if s.strip() == "":
            return ""
        is_primitive = Primitive.check_valid_primitive_name(s)
        if is_primitive:
            return s
        else:
            return "x0"
            #if s[0] == "x":
            #    return s
            #elif s[0] == "c":
            #    return s[:s.find(" ")]
            #elif s[0] == "e":
            #    return s[:s.find(" ")]
    num_primitives = tree.primitive_set().num_primitives()
    #num_features = tree.terminal_set().num_features()
    #num_constants = tree.terminal_set().num_constants()
    #num_ephemeral = tree.terminal_set().num_ephemeral()
    tot_attr = num_primitives + 1  # num_features + num_constants + num_ephemeral
    tot_attr_names = tree.primitive_set().primitive_names() + ["x0"]  # list(range(tree.terminal_set().num_features())) + list(range(tree.terminal_set().num_constants())) + list(range(tree.terminal_set().num_ephemeral()))
    dic = {"": [0.0]*tot_attr}
    t = 0
    for p in tot_attr_names:
        if t < num_primitives:
            curr_key = p
        else:
            curr_key = "x0"
        #elif num_primitives <= t < num_primitives + num_features:
        #    curr_key = "x" + str(p)
        #elif num_primitives + num_features <= t < num_primitives + num_features + num_constants:
        #    curr_key = "c" + str(p)
        #elif num_primitives + num_features + num_constants <= t < num_primitives + num_features + num_constants + num_ephemeral:
        #    curr_key = "e" + str(p)
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
    return np.array(arr, dtype=np.float32)


# ==============================================================================================================
# LABELS CALCULATOR
# ==============================================================================================================


# weights is a dictionary or list of dictionary while data is a list of primitive trees (this holds for methods of this section)
def build_dataset_counts_as_input_number_of_nodes_as_target(data):
    X = PrimitiveTree.extract_counting_features_from_list_of_trees(data)
    y = []
    for t in data:
        y.append(t.number_of_nodes())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset_counts_as_input_weights_average_as_target(data, weights):
    X = PrimitiveTree.extract_counting_features_from_list_of_trees(data)
    y = []
    for t in data:
        c = total_level_wise_weights_tree_converter(t, weights)
        s = c.sum()/float(t.number_of_nodes())
        y.append(s)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset_counts_as_input_handcraftedinterpretability_score_as_target(data):
    X = PrimitiveTree.extract_counting_features_from_list_of_trees(data)
    y = []
    for t in data:
        number_of_nodes = float(t.number_of_nodes())
        depth = float(t.depth())
        max_breadth = float(t.actual_max_breadth())
        max_degree = float(t.actual_max_degree())
        number_of_leaf_nodes = float(len(t.leaf_nodes()))
        number_of_internal_nodes = float(len(t.internal_nodes()))
        leaf_internal_nodes_ratio = number_of_leaf_nodes / number_of_internal_nodes
        leaf_nodes_perc = number_of_leaf_nodes / number_of_nodes
        degree_breadth_ratio = max_degree / max_breadth
        depth_number_of_nodes_ratio = depth / number_of_nodes
        s = depth_number_of_nodes_ratio + degree_breadth_ratio + leaf_nodes_perc
        y.append(s)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset_onehot_as_input_number_of_nodes_as_target(data):
    X, y = [], []
    for t in data:
        X.append(one_hot_tree(t))
        y.append(t.number_of_nodes())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset_onehot_as_input_weights_average_as_target(data, weights):
    X, y = [], []
    for t in data:
        c = total_level_wise_weights_tree_converter(t, weights)
        s = c.sum()/float(t.number_of_nodes())
        X.append(one_hot_tree(t))
        y.append(s)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset_onehot_as_input_handcraftedinterpretability_score_as_target(data):
    X, y = [], []
    for t in data:
        number_of_nodes = float(t.number_of_nodes())
        depth = float(t.depth())
        max_breadth = float(t.actual_max_breadth())
        max_degree = float(t.actual_max_degree())
        number_of_leaf_nodes = float(len(t.leaf_nodes()))
        number_of_internal_nodes = float(len(t.internal_nodes()))
        leaf_internal_nodes_ratio = number_of_leaf_nodes / number_of_internal_nodes
        leaf_nodes_perc = number_of_leaf_nodes / number_of_nodes
        degree_breadth_ratio = max_degree / max_breadth
        depth_number_of_nodes_ratio = depth / number_of_nodes
        s = depth_number_of_nodes_ratio + degree_breadth_ratio + leaf_nodes_perc
        X.append(one_hot_tree(t))
        y.append(s)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset_onehot_as_input_pwis_as_target(data, ranking, max_weight=1.0):
    X, y = [], []
    for t in data:
        s = t.compute_property_and_weights_based_interpretability_score(ranking, max_weight)
        X.append(one_hot_tree(t))
        y.append(s)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
