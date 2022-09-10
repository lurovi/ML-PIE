from genepro.node import Node
import random
import numpy as np
from exps.groundtruth.GroundTruthComputer import GroundTruthComputer

from nsgp.structure.TreeStructure import TreeStructure


class WeightsSumNegComputer(GroundTruthComputer):
    def __init__(self, structure: TreeStructure, seed: int = None):
        super().__init__()
        self.__seed = seed
        if self.__seed is not None:
            random.seed(self.__seed)
            np.random.seed(self.__seed)
        self.__structure = structure
        self.__operator_weights = [-abs(self.__structure.sample_operator_weight(i)) for i in range(self.__structure.get_number_of_operators())]
        self.__feature_weights = [-abs(self.__structure.sample_feature_weight(i)) for i in
                                   range(self.__structure.get_number_of_features())]
        self.__constant_weight = [-abs(self.__structure.sample_constant_weight()) for _ in range(1)]
        self.__weights = self.__operator_weights + self.__feature_weights + self.__constant_weight

    def compute(self, tree: Node) -> float:
        encoding = self.__structure.generate_encoding("counts", tree, False).tolist()
        if len(encoding) > self.__structure.get_size():
            encoding = encoding[:len(encoding)-3]
        return sum([encoding[i] * self.__weights[i] for i in range(len(encoding))])
