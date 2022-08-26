from typing import List, Dict

from genepro.variation import generate_random_tree

from genepro.node_impl import *
from genepro.node import Node

from genepro.util import one_hot_encode_tree, counts_encode_tree, counts_level_wise_encode_tree


class TreeGrammarStructure:
    def __init__(self, operators: List[Node], n_features: int,
                 max_depth: int, weights: List[List[float]] = None):
        self.__size = len(operators) + n_features + 1
        if weights is not None:
            if len(weights) != max_depth + 1:
                raise AttributeError(
                    "The number of rows in weights must be equal to the number of layers given by max_depth + 1.")
            for l in weights:
                if len(l) != self.__size:
                    raise AttributeError(
                        "Each row in weights must have a number of weights that is equal to num_operators + num_features + 1.")
            self.__weights = weights
        else:
            self.__weights = None
        self.__symbols = [str(op.symb) for op in operators]
        self.__operators = operators
        self.__n_operators = len(operators)
        self.__n_features = n_features
        self.__max_depth = max_depth
        self.__n_layers = max_depth + 1
        self.__max_arity = max([int(op.arity) for op in operators])
        self.__terminals = [Feature(i) for i in range(n_features)] + [Constant()]
        self.__n_terminals = len(self.__terminals)

    def get_weight(self, idx_layer: int, idx_op: int) -> float:
        if self.__weights is None:
            raise AttributeError("Cannot call this method because weights are currently None.")
        if not (0 <= idx_layer < self.get_number_of_layers()):
            raise IndexError(f"{idx_layer} is out of range as index of layers.")
        if not (0 <= idx_op < self.get_size()):
            raise IndexError(f"{idx_op} is out of size.")
        return self.__weights[idx_layer][idx_op]

    def get_symbol(self, idx: int) -> str:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of symbols.")
        return self.__symbols[idx]

    def get_operator(self, idx: int) -> Node:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of operators.")
        return self.__operators[idx]

    def get_feature(self, idx: int) -> Node:
        if not (0 <= idx < self.get_number_of_features()):
            raise IndexError(f"{idx} is out of range as index of features.")
        return self.__terminals[idx]

    def get_constant(self) -> Node:
        return self.__terminals[self.get_number_of_terminals()-1]

    def get_number_of_operators(self) -> int:
        return self.__n_operators

    def get_number_of_features(self) -> int:
        return self.__n_features

    def get_number_of_terminals(self) -> int:
        return self.__n_terminals

    def get_max_depth(self) -> int:
        return self.__max_depth

    def get_max_arity(self) -> int:
        return self.__max_arity

    def get_number_of_layers(self) -> int:
        return self.__n_layers

    def get_size(self) -> int:
        return self.__size

    def set_weights(self, weights: List[List[float]]) -> None:
        if len(weights) != self.get_max_depth() + 1:
            raise AttributeError(
                "The number of rows in weights must be equal to the number of layers given by max_depth + 1.")
        for l in weights:
            if len(l) != self.get_size():
                raise AttributeError(
                    "Each row in weights must have a number of weights that is equal to num_operators + num_features + 1.")
        self.__weights = weights

    def generate_tree(self) -> Node:
        t = generate_random_tree(self.__operators, self.__terminals, max_depth=self.get_max_depth(), curr_depth=0)
        # This call initializes empty ephemeral constants to valid floating-point values,
        # otherwise, this ephemeral remains equal to "const?" which is not a valid floating-point value.
        # t.get_readable_repr()
        return t

    def get_dict_representation(self, tree: Node) -> Dict[int, str]:
        return tree.get_dict_repr(self.get_max_arity())

    def generate_counts_encoding(self, tree: Node, additional_properties: bool = False) -> List[float]:

        return counts_encode_tree(tree, self.__symbols, self.get_number_of_features(), self.get_max_depth(),
                                  self.get_max_arity(), additional_properties)

    def generate_level_wise_counts_encoding(self, tree: Node, additional_properties: bool = False) -> List[float]:

        return counts_level_wise_encode_tree(tree, self.__symbols, self.get_number_of_features(), self.get_max_depth(),
                                             self.get_max_arity(), additional_properties)

    def generate_one_hot_encoding(self, tree: Node) -> List[float]:
        return one_hot_encode_tree(tree, self.__symbols, self.get_number_of_features(), self.get_max_depth(),
                                   self.get_max_arity())
