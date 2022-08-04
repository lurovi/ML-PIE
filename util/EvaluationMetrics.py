import numpy as np
from typing import Tuple, Dict, List, Any, Callable


class EvaluationMetrics:

    @staticmethod
    def model_accuracy(confusion_matrix: np.ndarray) -> Tuple[float, Dict[int, Dict[str, float]]]:
        N = confusion_matrix.shape[0]
        acc = sum([confusion_matrix[i, i] for i in range(N)])
        class_performance = {i: {} for i in range(N)}
        for i in range(N):
            positive_rate = confusion_matrix[i, i]
            true_positives = confusion_matrix[i, :].sum()
            predicted_positives = confusion_matrix[:, i].sum()
            class_performance[i]["precision"] = positive_rate / predicted_positives if predicted_positives != 0 else 0
            class_performance[i]["recall"] = positive_rate / true_positives if true_positives != 0 else 0
            class_performance[i]["f1"] = (2 * class_performance[i]["precision"] * class_performance[i]["recall"]) / (
                    class_performance[i]["precision"] + class_performance[i]["recall"]) if class_performance[i][
                                                                                               "precision"] + \
                                                                                           class_performance[i][
                                                                                               "recall"] != 0 else 0
        return acc / confusion_matrix.sum(), class_performance

    @staticmethod
    def spearman_footrule(origin: List[Any], estimated: List[Any], equality: Callable[[Any, Any], bool]) -> float:
        distance = 0
        for ie in range(len(origin)):
            exit_loop = False
            it = 0
            while not (exit_loop) and it < len(estimated):
                if equality(origin[ie], estimated[it]):
                    distance += abs(ie - it)
                    exit_loop = True
                it += 1
        n = len(origin)
        if n % 2 == 0:
            distance *= 3.0 / float(np.square(n))
        else:
            distance *= 3.0 / float(np.square(n) - 1.0)
        return distance
