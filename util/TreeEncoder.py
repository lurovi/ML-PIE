import re

import numpy as np
from typing import Any, List, Tuple

from genepro.node import Node


from nsgp.structure.TreeStructure import TreeGrammarStructure


class TreeEncoder:

    #########################################################################################################
    # ===================================== ENCODING WITH genepro ===========================================
    #########################################################################################################

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
            X.append(structure.generate_encoding("one_hot", t))
            y.append(t.get_n_nodes())
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_onehot_as_input_weights_sum_as_target(data: List[Node], structure: TreeGrammarStructure) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_encoding("one_hot", t))
            y.append(TreeEncoder.compute_ground_truth_as_weights_sum(t, structure))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_onehot_as_input_add_prop_as_target(data: List[Node], structure: TreeGrammarStructure) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            enc = structure.generate_encoding("counts", t)
            X.append(structure.generate_encoding("one_hot", t))
            y.append(sum(enc[-3:]))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_level_wise_counts_as_input_number_of_nodes_as_target(data: List[Node], structure: TreeGrammarStructure, scaler: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_encoding("level_wise_counts", t))
            y.append(t.get_n_nodes())
        X = np.array(X, dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        return X, np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_level_wise_counts_as_input_weights_sum_as_target(data: List[Node], structure: TreeGrammarStructure, scaler: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_encoding("level_wise_counts", t))
            y.append(TreeEncoder.compute_ground_truth_as_weights_sum(t, structure))
        X = np.array(X, dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        return X, np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_level_wise_counts_as_input_add_prop_as_target(data: List[Node], structure: TreeGrammarStructure, scaler: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            enc = structure.generate_encoding("level_wise_counts", t)
            X.append(enc)
            y.append(sum(enc[-3:]))
        X = np.array(X, dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        return X, np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_counts_as_input_number_of_nodes_as_target(data: List[Node], structure: TreeGrammarStructure, scaler: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_encoding("counts", t))
            y.append(t.get_n_nodes())
        X = np.array(X, dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        return X, np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_counts_as_input_weights_sum_as_target(data: List[Node], structure: TreeGrammarStructure, scaler: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            X.append(structure.generate_encoding("counts", t))
            y.append(TreeEncoder.compute_ground_truth_as_weights_sum(t, structure))
        X = np.array(X, dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        return X, np.array(y, dtype=np.float32)

    @staticmethod
    def create_dataset_counts_as_input_add_prop_as_target(data: List[Node], structure: TreeGrammarStructure, scaler: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for t in data:
            enc = structure.generate_encoding("counts", t)
            X.append(enc)
            y.append(sum(enc[-3:]))
        X = np.array(X, dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        return X, np.array(y, dtype=np.float32)
