from genepro.node import Node
import random
import numpy as np
from exps.groundtruth.GroundTruthComputer import GroundTruthComputer

from nsgp.structure.TreeStructure import TreeStructure


class NodeWiseWeightsSumNegComputer(GroundTruthComputer):
    def __init__(self, structure: TreeStructure, seed: int = None, idx: int = 1):
        super().__init__()
        self.__idx = idx
        self.set_name("node_wise_weights_sum"+"_"+str(idx))
        self.__seed = seed
        if self.__seed is not None:
            random.seed(self.__seed)
            np.random.seed(self.__seed)
        self.__structure = structure
        self.__weights = []
        for _ in range(self.__structure.get_max_n_nodes()):
            self.__operator_weights = [-abs(self.__structure.sample_operator_weight(i)) for i in range(self.__structure.get_number_of_operators())]
            self.__feature_weights = [-abs(self.__structure.sample_feature_weight(i)) for i in range(self.__structure.get_number_of_features())]
            self.__constant_weight = [-abs(self.__structure.sample_constant_weight()) for _ in range(1)]
            self.__weights.extend(self.__operator_weights + self.__feature_weights + self.__constant_weight)

    def compute(self, tree: Node) -> float:
        encoding = self.__structure.generate_encoding("one_hot", tree, False).tolist()
        return sum([encoding[i] * self.__weights[i] for i in range(len(encoding))])
