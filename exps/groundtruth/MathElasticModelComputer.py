from genepro.node import Node
from typing import List

from genepro.util import compute_linear_model_discovered_in_math_formula_interpretability_paper

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer


class MathElasticModelComputer(GroundTruthComputer):
    def __init__(self, difficult_operators: List[str] = None):
        super().__init__()
        self.__difficult_operators = difficult_operators

    def compute(self, tree: Node) -> float:
        return compute_linear_model_discovered_in_math_formula_interpretability_paper(tree, self.__difficult_operators)
