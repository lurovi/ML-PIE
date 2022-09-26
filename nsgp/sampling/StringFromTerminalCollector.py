from nsgp.sampling.FeedbackCollector import FeedbackCollector
from typing import List, Tuple
from genepro.node import Node


class StringFromTerminalCollector(FeedbackCollector):
    def __init__(self):
        super().__init__()

    def collect_feedback(self, pairs: List[Tuple[Node, Node]]) -> List[int]:
        feedbacks = []
        print("")
        for first_tree, second_tree in pairs:
            print("=============================================================")
            print("")
            print("CURRENT PAIR:")
            print("")
            print(first_tree.get_string_as_lisp_expr())
            print("_____________________________________________________________")
            print("")
            print(second_tree.get_string_as_lisp_expr())
            print("")
            print("Choose the most interpretable tree according to you.")
            print("")
            label = int(input("Type 1 for choosing the first tree. Type 2 for choosing the second tree: "))
            print("")
            if label == 1:
                curr_feedback = -1
            elif label == 2:
                curr_feedback = 1
            else:
                raise ValueError(f"{label} is not a valid input. Choose either 1 or 2.")
            feedbacks.append(curr_feedback)
            print("=============================================================")
            print("")
        return feedbacks
