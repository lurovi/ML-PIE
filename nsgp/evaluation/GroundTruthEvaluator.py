from genepro.node import Node

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from nsgp.evaluation.TreeEvaluator import TreeEvaluator


class GroundTruthEvaluator(TreeEvaluator):
    def __init__(self, ground_truth_computer: GroundTruthComputer, negate: bool = False):
        super().__init__()
        self.__ground_truth_computer = ground_truth_computer
        self.__sign = -1.0 if negate else 1.0

    def evaluate(self, tree: Node, **kwargs) -> float:
        return self.__sign * self.__ground_truth_computer.compute(tree)
