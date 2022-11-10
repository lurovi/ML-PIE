from typing import List, Tuple

from genepro.util import compute_linear_scaling
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
            mse: float = np.square(np.core.umath.clip(res_val - validation_labels, -1e+20, 1e+20)).sum() / float(len(validation_labels))
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
    def apply_tree_to_test_set_for_r2_score(test_set: np.ndarray, test_labels: np.ndarray, tree: Node, slope: float, intercept: float) -> float:
        res_test = np.core.umath.clip(tree(test_set), -1e+10, 1e+10)
        slope = np.core.umath.clip(slope, -1e+10, 1e+10)
        intercept = np.core.umath.clip(intercept, -1e+10, 1e+10)
        res_test = intercept + np.core.umath.clip(slope * res_test, -1e+10, 1e+10)
        res_test = np.core.umath.clip(res_test, -1e+10, 1e+10)
        return r2_score(test_labels, res_test)

    @staticmethod
    def find_slope_intercept_training(training_set: np.ndarray, training_labels: np.ndarray, tree: Node, linear_scaling: bool = True) -> Tuple[np.ndarray, float, float, float]:
        res: np.ndarray = np.core.umath.clip(tree(training_set), -1e+10, 1e+10)
        slope, intercept = 1.0, 0.0
        if linear_scaling:
            slope, intercept = compute_linear_scaling(training_labels, res)
            slope = np.core.umath.clip(slope, -1e+10, 1e+10)
            intercept = np.core.umath.clip(intercept, -1e+10, 1e+10)
            res = intercept + np.core.umath.clip(slope * res, -1e+10, 1e+10)
            res = np.core.umath.clip(res, -1e+10, 1e+10)
        mse: float = np.square(np.core.umath.clip(res - training_labels, -1e+20, 1e+20)).sum() / float(len(training_labels))
        if mse > 1e+20:
            mse = 1e+20
        return res, mse, slope, intercept

    @staticmethod
    def predict_validation_data(validation_set: np.ndarray, validation_labels: np.ndarray, tree: Node, slope: float, intercept: float) -> Tuple[np.ndarray, float]:
        res: np.ndarray = np.core.umath.clip(tree(validation_set), -1e+10, 1e+10)
        res = intercept + np.core.umath.clip(slope * res, -1e+10, 1e+10)
        res = np.core.umath.clip(res, -1e+10, 1e+10)
        mse: float = np.square(np.core.umath.clip(res - validation_labels, -1e+20, 1e+20)).sum() / float(len(validation_labels))
        if mse > 1e+20:
            mse = 1e+20
        return res, mse
