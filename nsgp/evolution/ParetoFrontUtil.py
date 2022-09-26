from typing import List

from pymoo.core.result import Result
import numpy as np
from sklearn.metrics import r2_score

from genepro.node import Node


class ParetoFrontUtil:

    @staticmethod
    def apply_validation_set_to_bi_objective_pareto_front(res: Result, validation_set: np.ndarray, validation_labels: np.ndarray) -> List:
        pareto_front = res.X
        new_pareto_front = []
        for p in pareto_front:
            tree = p[0]
            res_val = tree(validation_set)
            mse: float = np.square(np.clip(res_val - validation_labels, -1e+20, 1e+20)).sum() / float(len(validation_labels))
            if mse > 1e+20:
                mse = 1e+20
            new_pareto_front.append((tree, mse, tree.get_n_nodes()))

        new_custom_front = []
        for i in range(len(new_pareto_front)):
            dominated = False
            for j in range(len(new_pareto_front)):
                if i != j and ((new_pareto_front[i][1] == new_pareto_front[j][1] and new_pareto_front[i][2] > new_pareto_front[j][2])
                               or (new_pareto_front[i][1] > new_pareto_front[j][1] and new_pareto_front[i][2] == new_pareto_front[j][2])
                               or (new_pareto_front[i][1] > new_pareto_front[j][1] and new_pareto_front[i][2] > new_pareto_front[j][2])):
                    dominated = True
            if not dominated:
                new_custom_front.append(new_pareto_front[i])
        return new_custom_front

    @staticmethod
    def apply_tree_to_test_set_for_r2_score(test_set: np.ndarray, test_labels: np.ndarray, tree: Node) -> float:
        res_test = tree(test_set)
        return r2_score(res_test, test_labels)
