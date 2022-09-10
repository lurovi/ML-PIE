from genepro.node import Node
from typing import List

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from nsgp.structure.TreeStructure import TreeStructure


class MathElasticModelComputer(GroundTruthComputer):
    def __init__(self, difficult_operators: List[str] = None):
        super().__init__()
        self.__difficult_operators = difficult_operators

    def compute(self, tree: Node) -> float:
        return TreeStructure.calculate_linear_model_discovered_in_math_formula_interpretability_paper(tree, self.__difficult_operators)
