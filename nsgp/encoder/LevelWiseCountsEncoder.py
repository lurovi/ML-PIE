import random

import numpy as np
from genepro.util import counts_level_wise_encode_tree
from sklearn.preprocessing import MinMaxScaler

from genepro.node import Node
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.structure.TreeStructure import TreeStructure


class LevelWiseCountsEncoder(TreeEncoder):
    def __init__(self, structure: TreeStructure, additional_properties: bool = True, seed: int = None):
        super().__init__()
        self.__structure = structure
        self.__seed = seed
        if self.__seed is not None:
            random.seed(self.__seed)
            np.random.seed(self.__seed)
        self.__additional_properties: bool = additional_properties
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = [self.__structure.generate_tree() for _ in range(10 ** 5)]
        data = [self.encode(t, False) for t in data]
        scaler.fit(np.array(data))
        self.set_scaler(scaler)
        self.set_name("level_wise_counts")

    def encode(self, tree: Node, apply_scaler: bool = True) -> np.ndarray:
        a = np.array(counts_level_wise_encode_tree(tree,
                                               [self.__structure.get_symbol(i) for i in range(self.__structure.get_number_of_operators())],
                                               self.__structure.get_number_of_features(),
                                               self.__structure.get_max_depth(),
                                               self.__additional_properties))
        if apply_scaler:
            a = self.scale(a)
        return a

    def size(self) -> int:
        return int((self.__structure.get_number_of_operators() + self.__structure.get_number_of_features() + 1) * self.__structure.get_number_of_layers()) + (3 if self.__additional_properties else 0)
