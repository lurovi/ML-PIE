from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from nsgp.sampling.FeedbackCollector import FeedbackCollector
from typing import List, Tuple
from genepro.node import Node


class GroundTruthCollector(FeedbackCollector):
    def __init__(self, ground_truth_computer: GroundTruthComputer):
        super().__init__()
        self.__ground_truth_computer = ground_truth_computer

    def collect_feedback(self, pairs: List[Tuple[Node, Node]]) -> List[float]:
        feedbacks = []
        for first_tree, second_tree in pairs:
            first_label = self.__ground_truth_computer.compute(first_tree)
            second_label = self.__ground_truth_computer.compute(second_tree)
            if first_label >= second_label:
                curr_feedback = -1
            else:
                curr_feedback = 1
            feedbacks.append(curr_feedback)
        return feedbacks
