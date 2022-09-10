from typing import List, Dict, Callable, Tuple, Any

from genepro.variation import generate_random_tree, safe_subtree_mutation, safe_subtree_crossover_two_children

from genepro.node_impl import *
from genepro.node import Node

from genepro.util import compute_linear_model_discovered_in_math_formula_interpretability_paper
from nsgp.encoder.TreeEncoder import TreeEncoder

from copy import deepcopy


class TreeStructure:
    def __init__(self, operators: List[Node],
                 n_features: int,
                 max_depth: int,
                 constants: List[Constant] = None,
                 ephemeral_func: Callable = None,
                 normal_distribution_parameters: List[Tuple[float, float]] = None):
        self.__size: int = len(operators) + n_features + 1
        if normal_distribution_parameters is not None:
            if len(normal_distribution_parameters) != self.__size:
                raise AttributeError("The number of elements in normal distribution parameters must be equal to size (num_operators + num_features + 1).")
            self.__normal_distribution_parameters: List[Tuple[float, float]] = deepcopy(normal_distribution_parameters)
        else:
            self.__normal_distribution_parameters: List[Tuple[float, float]] = None
        self.__symbols: List[str] = [str(op.symb) for op in operators]
        self.__operators: List[Node] = deepcopy(operators)
        self.__n_operators: int = len(operators)
        self.__n_features: int = n_features
        self.__features: List[Feature] = [Feature(i) for i in range(n_features)]
        self.__max_depth: int = max_depth
        self.__n_layers: int = max_depth + 1
        self.__max_arity: int = max([int(op.arity) for op in operators])
        self.__max_n_nodes: int = int((self.__max_arity ** self.__n_layers - 1)/float(self.__max_arity - 1))
        self.__constants: List[Constant] = []
        if constants is not None:
            self.__constants = deepcopy(constants)
        self.__ephemeral_func: Callable = ephemeral_func
        self.__n_constants: int = len(self.__constants)
        self.__terminals: List[Node] = self.__features + self.__constants
        self.__n_terminals: int = len(self.__terminals) + (1 if self.__ephemeral_func is not None else 0)

        self.__encoding_func_dict: Dict = {}

    def get_encoding_type_strings(self) -> List[str]:
        return list(self.__encoding_func_dict.keys())

    def get_normal_distribution_parameters(self) -> List[Tuple[float, float]]:
        if self.__normal_distribution_parameters is None:
            raise ValueError("Normal distribution parameters have not been set yet.")
        return deepcopy(self.__normal_distribution_parameters)

    def set_normal_distribution_parameters(self, normal_distribution_parameters: List[Tuple[float, float]] = None) -> None:
        if normal_distribution_parameters is not None:
            if len(normal_distribution_parameters) != self.__size:
                raise AttributeError("The number of elements in normal distribution parameters must be equal to size (num_operators + num_features + 1).")
            self.__normal_distribution_parameters: List[Tuple[float, float]] = deepcopy(normal_distribution_parameters)
        else:
            self.__normal_distribution_parameters: List[Tuple[float, float]] = None

    def __sample_weight(self, idx: int) -> float:
        if not (0 <= idx < self.__size):
            raise IndexError(f"{idx} is out of range as size.")
        return np.random.normal(self.__normal_distribution_parameters[idx][0], self.__normal_distribution_parameters[idx][1])

    def sample_operator_weight(self, idx: int) -> float:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of operators.")
        return self.__sample_weight(idx)

    def sample_feature_weight(self, idx: int) -> float:
        if not (0 <= idx < self.get_number_of_features()):
            raise IndexError(f"{idx} is out of range as index of features.")
        return self.__sample_weight(self.get_number_of_operators() + idx)

    def sample_constant_weight(self) -> float:
        return self.__sample_weight(self.__size - 1)

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

    def get_max_n_nodes(self) -> int:
        return self.__max_n_nodes

    def get_number_of_layers(self) -> int:
        return self.__n_layers

    def get_size(self) -> int:
        return self.__size

    def generate_tree(self) -> Node:
        return generate_random_tree(self.__operators, self.__terminals, max_depth=self.get_max_depth(),
                                    curr_depth=0, ephemeral_func=self.__ephemeral_func)

    def safe_subtree_mutation(self, tree: Node) -> Node:
        return safe_subtree_mutation(tree, self.__operators, self.__terminals, max_depth=self.__max_depth,
                                     ephemeral_func=self.__ephemeral_func)

    def safe_subtree_crossover_two_children(self, tree_1: Node, tree_2: Node) -> Tuple[Node, Node]:
        return safe_subtree_crossover_two_children(tree_1, tree_2, max_depth=self.__max_depth)

    def get_dict_representation(self, tree: Node) -> Dict[int, str]:
        return tree.get_dict_repr(self.get_max_arity())

    def register_encoder(self, encoder: TreeEncoder) -> None:
        if self != encoder.get_structure():
            raise AttributeError(f"The input tree encoder has a tree structure that is different from current tree structure (self).)")
        if encoder.get_name() in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoder.get_name()} already exists as key of the dictionary of encodings in this tree structure.")
        self.__encoding_func_dict[encoder.get_name()] = {"encode": encoder.encode, "scaler": encoder.get_scaler(), "scale": encoder.scale, "size": encoder.size()}

    def register_encoders(self, encoders: List[TreeEncoder]) -> None:
        names = []
        for e in encoders:
            names.append(e.get_name())
            if self != e.get_structure():
                raise AttributeError(f"The input tree encoder has a tree structure that is different from current tree structure (self).)")
            if e.get_name() in self.__encoding_func_dict.keys():
                raise AttributeError(f"{e.get_name()} already exists as key of the dictionary of encodings in this tree structure.")
        if len(names) != len(list(set(names))):
            raise AttributeError(f"Names of the input encoders must all be distinct.")
        for e in encoders:
            self.register_encoder(e)

    def unregister_encoder(self, encoding_type: str) -> None:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        self.__encoding_func_dict.pop(encoding_type)

    def generate_encoding(self, encoding_type: str, tree: Node, apply_scaler: bool = True) -> np.ndarray:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type]["encode"](tree, apply_scaler)

    def scale_encoding(self, encoding_type: str, encoding: np.ndarray) -> np.ndarray:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type]["scale"](encoding)

    def get_scaler_on_encoding(self, encoding_type: str) -> Any:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type]["scaler"]

    def get_encoding_size(self, encoding_type: str) -> int:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type]["size"]

    @staticmethod
    def calculate_linear_model_discovered_in_math_formula_interpretability_paper(tree: Node,
                                                                                 difficult_operators: List[str] = None) -> float:
        return compute_linear_model_discovered_in_math_formula_interpretability_paper(tree, difficult_operators)
