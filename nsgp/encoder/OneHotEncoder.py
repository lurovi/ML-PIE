import numpy as np
from sklearn.pipeline import Pipeline

from genepro.util import one_hot_encode_tree

from genepro.node import Node
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.structure.TreeStructure import TreeStructure


class OneHotEncoder(TreeEncoder):
    def __init__(self, structure: TreeStructure):
        super().__init__()
        self.__structure = structure
        scaler = Pipeline(steps=[("do_nothing_scaler", None)])
        data = [self.__structure.generate_tree() for _ in range(5)]
        data = [self.encode(t, False) for t in data]
        scaler.fit(np.array(data))
        self.set_scaler(scaler)
        self.set_name("one_hot")

    def encode(self, tree: Node, apply_scaler: bool = True) -> np.ndarray:
        a = np.array(one_hot_encode_tree(tree,
                                     [self.__structure.get_symbol(i) for i in range(self.__structure.get_number_of_operators())],
                                     self.__structure.get_number_of_features(),
                                     self.__structure.get_max_depth(),
                                     self.__structure.get_max_arity()))
        return a

    def size(self) -> int:
        return int((self.__structure.get_number_of_operators() + self.__structure.get_number_of_features() + 1) * self.__structure.get_max_n_nodes())
