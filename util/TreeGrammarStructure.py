from typing import List, Dict, Callable, Tuple

from genepro.variation import generate_random_tree, safe_subtree_mutation, safe_subtree_crossover_two_children

from genepro.node_impl import *
from genepro.node import Node

from genepro.util import one_hot_encode_tree, counts_encode_tree, counts_level_wise_encode_tree, counts_level_wise_encode_tree_fast


class TreeGrammarStructure:
    def __init__(self, operators: List[Node], n_features: int,
                 max_depth: int, weights: List[List[float]] = None, constants: List[Constant] = None, ephemeral_func: Callable = None):
        self.__size: int = len(operators) + n_features + 1
        if weights is not None:
            if len(weights) != max_depth + 1:
                raise AttributeError(
                    "The number of rows in weights must be equal to the number of layers given by max_depth + 1.")
            for l in weights:
                if len(l) != self.__size:
                    raise AttributeError(
                        "Each row in weights must have a number of weights that is equal to num_operators + num_features + 1.")
            self.__weights: List[List[float]] = weights
        else:
            self.__weights: List[List[float]] = None
        self.__symbols: List[str] = [str(op.symb) for op in operators]
        self.__operators: List[Node] = operators
        self.__n_operators: int = len(operators)
        self.__n_features: int = n_features
        self.__features: List[Feature] = [Feature(i) for i in range(n_features)]
        self.__max_depth: int = max_depth
        self.__n_layers: int = max_depth + 1
        self.__max_arity: int = max([int(op.arity) for op in operators])
        self.__constants: List[Constant] = []
        if constants is not None:
            self.__constants = constants
        self.__ephemeral_func: Callable = ephemeral_func
        self.__n_constants: int = len(self.__constants)
        self.__terminals: List[Node] = self.__features + self.__constants
        self.__n_terminals: int = len(self.__terminals) + (1 if self.__ephemeral_func is not None else 0)

    def get_weight(self, idx_layer: int, idx_op: int) -> float:
        if self.__weights is None:
            raise AttributeError("Cannot call this method because weights are currently None.")
        if not (0 <= idx_layer < self.get_number_of_layers()):
            raise IndexError(f"{idx_layer} is out of range as index of layers.")
        if not (0 <= idx_op < self.get_size()):
            raise IndexError(f"{idx_op} is out of size.")
        return self.__weights[idx_layer][idx_op]

    def set_weights(self, weights: List[List[float]]) -> None:
        if len(weights) != self.get_max_depth() + 1:
            raise AttributeError(
                "The number of rows in weights must be equal to the number of layers given by max_depth + 1.")
        for l in weights:
            if len(l) != self.get_size():
                raise AttributeError(
                    "Each row in weights must have a number of weights that is equal to num_operators + num_features + 1.")
        self.__weights = weights

    def get_symbol(self, idx: int) -> str:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of symbols.")
        return self.__symbols[idx]

    def get_operator(self, idx: int) -> Node:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of operators.")
        return self.__operators[idx]

    def get_feature(self, idx: int) -> Feature:
        if not (0 <= idx < self.get_number_of_features()):
            raise IndexError(f"{idx} is out of range as index of features.")
        return self.__features[idx]

    def get_constant(self, idx: int) -> Constant:
        if not (0 <= idx < self.get_number_of_constants()):
            raise IndexError(f"{idx} is out of range as index of constants.")
        return self.__constants[idx]

    def sample_ephemeral_random_constant(self) -> float:
        if self.__ephemeral_func is None:
            raise AttributeError("Ephemeral function has not been defined in the constructor of this instance.")
        return self.__ephemeral_func()

    def get_number_of_operators(self) -> int:
        return self.__n_operators

    def get_number_of_features(self) -> int:
        return self.__n_features

    def get_number_of_constants(self) -> int:
        return self.__n_constants

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

    def generate_tree(self) -> Node:
        return generate_random_tree(self.__operators, self.__terminals, max_depth=self.get_max_depth(),
                                    curr_depth=0, ephemeral_func=self.__ephemeral_func)

    def safe_subtree_mutation(self, tree: Node) -> Node:
        return safe_subtree_mutation(tree, self.__operators, self.__terminals, max_depth=self.__max_depth, ephemeral_func=self.__ephemeral_func)

    def safe_subtree_crossover_two_children(self, tree_1: Node, tree_2: Node) -> Tuple[Node, Node]:
        return safe_subtree_crossover_two_children(tree_1, tree_2, max_depth=self.__max_depth)

    def get_dict_representation(self, tree: Node) -> Dict[int, str]:
        return tree.get_dict_repr(self.get_max_arity())

    def generate_counts_encoding(self, tree: Node, additional_properties: bool = False) -> List[float]:

        return counts_encode_tree(tree, self.__symbols, self.get_number_of_features(), self.get_max_depth(),
                                  self.get_max_arity(), additional_properties)

    def generate_level_wise_counts_encoding(self, tree: Node, additional_properties: bool = False) -> List[float]:

        return counts_level_wise_encode_tree(tree, self.__symbols, self.get_number_of_features(), self.get_max_depth(),
                                             self.get_max_arity(), additional_properties)

    def generate_level_wise_counts_encoding_fast(self, tree: Node, additional_properties: bool = False) -> List[float]:

        return counts_level_wise_encode_tree_fast(tree, self.__symbols, self.get_number_of_features(), self.get_max_depth(),
                                                  self.get_max_arity(), additional_properties)

    def generate_one_hot_encoding(self, tree: Node) -> List[float]:
        return one_hot_encode_tree(tree, self.__symbols, self.get_number_of_features(), self.get_max_depth(),
                                   self.get_max_arity())
