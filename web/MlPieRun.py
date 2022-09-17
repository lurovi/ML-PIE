import time

from sympy import latex

from genepro.node import Node
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from threads import OptimizationThread

import numpy as np

import pandas as pd

path = "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\results\\"


class MlPieRun:
    def __init__(self, run_id, optimization_thread, interpretability_estimate_updater):
        self.timeout_time = 3 * 60
        self.run_id = run_id
        self.optimization_thread: OptimizationThread = optimization_thread
        self.interpretability_estimate_updater: InterpretabilityEstimateUpdater = interpretability_estimate_updater
        self.feedback_counter: int = -1
        self.feedback_duration: list[float] = []
        self.feedback_requests: list[tuple] = []
        self.encoded_requests: list[np.ndarray] = []
        self.feedback_responses: list[int] = []
        self.feedback_request_time: float = 0
        self.feedback_requests_iterations: list[int] = []
        self.feedback_responses_iterations: list[int] = []

    def start(self) -> None:
        self.optimization_thread.start()

    def request_trees(self) -> dict:
        if not self.optimization_thread.is_alive():
            self.flush()
            return {}
        # if the previous request was not answered I give it back again
        # might change it to -> sample again without incrementing the counter
        if len(self.feedback_requests) > len(self.feedback_responses):
            return {
                't1': self.feedback_requests[self.feedback_counter][0],
                't2': self.feedback_requests[self.feedback_counter][1],
                'it': self.feedback_requests_iterations[self.feedback_counter]
            }
        self.feedback_counter += 1
        requested_values = self.interpretability_estimate_updater.request_trees()
        iteration = self.optimization_thread.get_current_iteration()
        self.feedback_requests_iterations.append(iteration)
        trees = (self.tree_to_latex(requested_values["t1"]), self.tree_to_latex(requested_values["t2"]))
        self.feedback_requests.append(trees)
        self.encoded_requests.append(requested_values["encoding"])
        self.feedback_request_time = time.time()
        total_generations = self.optimization_thread.termination[1]
        return {'t1': trees[0], 't2': trees[1], 'it': iteration, 'progress': 100 * iteration / total_generations}

    def provide_feedback(self, feedback: int) -> bool:
        if not self.optimization_thread.is_alive():
            self.flush()
            return False
        if len(self.feedback_requests) == len(self.feedback_responses):
            return True
        self.feedback_duration.append(time.time() - self.feedback_request_time)
        self.feedback_responses.append(feedback)
        self.feedback_responses_iterations.append(self.optimization_thread.get_current_iteration())
        self.interpretability_estimate_updater.provide_feedback(
            encoded_trees=self.encoded_requests[self.feedback_counter],
            feedback=feedback
        )
        return True

    def flush(self) -> None:
        self.optimization_thread.join()
        tree_1, tree_2 = zip(*self.feedback_requests)
        feedback_data = pd.DataFrame(list(zip(
            self.feedback_duration, tree_1, tree_2, self.encoded_requests, self.feedback_responses,
            self.feedback_requests_iterations, self.feedback_responses_iterations)),
            columns=['duration', 'tree_1', 'tree_2', 'encoding', 'feedback', 'req_iteration', 'resp_iteration'])
        feedback_data.to_csv(path_or_buf=path + "feedback-" + self.run_id + ".csv")

    def is_abandoned(self) -> bool:
        return time.time() - self.feedback_request_time > self.timeout_time

    @staticmethod
    def tree_to_latex(tree: Node) -> str:
        readable_repr = tree.get_readable_repr().replace("u-", "-")
        latex_repr = latex(readable_repr)
        return latex_repr
