import re

import numpy as np
from typing import Callable, Any, Iterable, Dict, List, Tuple

from genepro.node import Node

from gp.tree.PrimitiveTree import PrimitiveTree
from gp.tree.Primitive import Primitive

from nsgp.TreeGrammarStructure import TreeGrammarStructure


class TreeEncoder:

    @staticmethod
    def compute_ground_truth_as_number_of_nodes(tree: Node) -> float:
        return float(tree.get_n_nodes())

    @staticmethod
    def compute_ground_truth_as_weights_sum(tree: Node, structure: TreeGrammarStructure) -> float:
        dictionary_encoded_tree = structure.get_dict_representation(tree)
        curr_node_index = 0
        total_sum = 0.0
        operators = [structure.get_symbol(i) for i in range(structure.get_number_of_operators())]
        for curr_layer in range(structure.get_number_of_layers()):
            curr_n_nodes = structure.get_max_arity() ** curr_layer
            for _ in range(curr_n_nodes):
                if curr_node_index in dictionary_encoded_tree:
                    node_content = dictionary_encoded_tree[curr_node_index]
                    if node_content in operators:
                        total_sum += structure.get_weight(curr_layer, operators.index(node_content))
                    elif node_content.startswith("x_"):
                        feature_index = int(node_content[2:])
                        if feature_index < structure.get_number_of_features():
                            total_sum += structure.get_weight(curr_layer, len(operators) + feature_index)
                        else:
                            raise Exception("More features than declared.")
                    elif re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$', node_content) or isinstance(node_content,
                                                                                                     float) or isinstance(
                            node_content, int):
                        total_sum += structure.get_weight(curr_layer, structure.get_size() - 1)
                    else:
                        raise Exception(f"Unexpected node content: {str(node_content)}.")
                curr_node_index += 1
        return total_sum

    @staticmethod
    def create_dataset_onehot_as_input_number_of_nodes_as_target(data: List[Node], structure: TreeGrammarStructure) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_one_hot_encoding(t))
            y.append(t.get_n_nodes())
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_onehot_as_input_weights_sum_as_target(data: List[Node], structure: TreeGrammarStructure) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_one_hot_encoding(t))
            y.append(TreeEncoder.compute_ground_truth_as_weights_sum(t, structure))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_counts_as_input_number_of_nodes_as_target(data: List[Node], structure: TreeGrammarStructure, scaler: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_counts_encoding(t, True))
            y.append(t.get_n_nodes())
        X = np.array(X, dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        return X, np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_counts_as_input_weights_sum_as_target(data: List[Node], structure: TreeGrammarStructure, scaler: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_counts_encoding(t, True))
            y.append(TreeEncoder.compute_ground_truth_as_weights_sum(t, structure))
        X = np.array(X, dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        return X, np.array(y, dtype=np.float32)

    @staticmethod
    def create_scaler_on_counts(structure: TreeGrammarStructure, base_scaler: Any, data: List[Node]) -> Any:
        data = [structure.generate_counts_encoding(t, True) for t in data]
        base_scaler.fit(np.array(data))
        return base_scaler

    @staticmethod
    def total_level_wise_weights_tree_converter(tree: PrimitiveTree, weights: List[Dict[str, float]]) -> np.ndarray:
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
                if s[0] == "x":
                    return s
                else:
                    return "c0"
                #elif s[0] == "c":
                #    return s[:s.find(" ")]
                #elif s[0] == "e":
                #    return s[:s.find(" ")]
        arr = []
        for layer_ind in range(tree.max_depth()):
            curr_layer = tree.layer(layer_ind)
            curr_weights = weights[layer_ind]
            # curr_layer = [0.0 if n == "" else curr_weights[__get_right_key_from_terminal(n)] for n in curr_layer]
            curr_layer = [0.0 if not(Primitive.check_valid_primitive_name(n)) else curr_weights[n] for n in curr_layer]
            arr.extend(curr_layer)
        return np.array(arr, dtype=np.float32)

    @staticmethod
    def one_hot_tree(tree: PrimitiveTree) -> np.ndarray:
        def __get_right_key_from_terminal(s: str) -> str:
            if s.strip() == "":
                return ""
            is_primitive = Primitive.check_valid_primitive_name(s)
            if is_primitive:
                return s
            else:
                if s[0] == "x":
                    return s
                else:
                    return "c0"
                # elif s[0] == "c":
                #    return s[:s.find(" ")]
                # elif s[0] == "e":
                #    return s[:s.find(" ")]
        num_primitives = tree.primitive_set().num_primitives()
        num_features = tree.terminal_set().num_features()
        tot_attr = num_primitives + num_features + 1
        tot_attr_names = tree.primitive_set().primitive_names() + list(range(tree.terminal_set().num_features())) + ["c0"]  # list(range(tree.terminal_set().num_constants())) + list(range(tree.terminal_set().num_ephemeral()))
        dic = {"": [0.0]*tot_attr}
        t = 0
        for p in tot_attr_names:
            if t < num_primitives:
                curr_key = p
            elif num_primitives <= t < num_primitives + num_features:
                curr_key = "x" + str(p)
            #elif num_primitives + num_features <= t < num_primitives + num_features + num_constants:
            #    curr_key = "c" + str(p)
            #elif num_primitives + num_features + num_constants <= t < num_primitives + num_features + num_constants + num_ephemeral:
            #    curr_key = "e" + str(p)
            else:
                curr_key = "c0"
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

    # weights is a dictionary or list of dictionary while data is a list of primitive trees (this holds for methods of this section)
    @staticmethod
    def build_dataset_counts_as_input_number_of_nodes_as_target(data):
        X = PrimitiveTree.extract_counting_features_from_list_of_trees(data)
        y = []
        for t in data:
            y.append(t.number_of_nodes())
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def build_dataset_counts_as_input_weights_sum_as_target(data, weights):
        X = PrimitiveTree.extract_counting_features_from_list_of_trees(data)
        y = []
        for t in data:
            c = TreeEncoder.total_level_wise_weights_tree_converter(t, weights)
            s = c.sum()
            y.append(s)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def build_dataset_counts_as_input_weights_average_as_target(data, weights):
        X = PrimitiveTree.extract_counting_features_from_list_of_trees(data)
        y = []
        for t in data:
            c = TreeEncoder.total_level_wise_weights_tree_converter(t, weights)
            s = c.sum()/float(len(t.internal_nodes())) + 1.0/float(t.number_of_nodes())
            y.append(s)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
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

    @staticmethod
    def build_dataset_onehot_as_input_number_of_nodes_as_target(data):
        X, y = [], []
        for t in data:
            X.append(TreeEncoder.one_hot_tree(t))
            y.append(t.number_of_nodes())
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def build_dataset_onehot_as_input_weights_sum_as_target(data, weights):
        X, y = [], []
        for t in data:
            c = TreeEncoder.total_level_wise_weights_tree_converter(t, weights)
            s = c.sum()
            X.append(TreeEncoder.one_hot_tree(t))
            y.append(s)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def build_dataset_onehot_as_input_weights_average_as_target(data, weights):
        X, y = [], []
        for t in data:
            c = TreeEncoder.total_level_wise_weights_tree_converter(t, weights)
            s = c.sum()/float(len(t.internal_nodes())) + 1.0/float(t.number_of_nodes())
            X.append(TreeEncoder.one_hot_tree(t))
            y.append(s)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
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
            X.append(TreeEncoder.one_hot_tree(t))
            y.append(s)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def build_dataset_onehot_as_input_pwis_as_target(data, ranking, max_weight=1.0):
        X, y = [], []
        for t in data:
            s = t.compute_property_and_weights_based_interpretability_score(ranking, max_weight)
            X.append(TreeEncoder.one_hot_tree(t))
            y.append(s)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
