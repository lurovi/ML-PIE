import time

from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from threads import OptimizationThread

import numpy as np

import pandas as pd

path = "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\results\\"


class MlPieRun:
    def __init__(self, run_id, optimization_thread, interpretability_estimate_updater):
        self.run_id = run_id
        self.optimization_thread: OptimizationThread = optimization_thread
        self.interpretability_estimate_updater: InterpretabilityEstimateUpdater = interpretability_estimate_updater
        self.feedback_counter: int = -1
        self.feedback_duration: list[float] = []
        self.feedback_requests: list[tuple] = []
        self.encoded_requests: list[np.ndarray] = []
        self.feedback_responses: list[int] = []
        self.feedback_request_time: float = 0

    def start(self) -> None:
        self.optimization_thread.start()

    def request_trees(self) -> tuple:
        if not self.optimization_thread.is_alive():
            self.flush()
            return ()
        # if the previous request was not answered I give it back again
        # might change it to -> sample again without incrementing the counter
        if len(self.feedback_requests) > len(self.feedback_responses):
            return self.feedback_requests[self.feedback_counter]
        self.feedback_counter += 1
        requested_values = self.interpretability_estimate_updater.request_trees()
        trees = (requested_values["t1"].get_string_as_lisp_expr(), requested_values["t2"].get_string_as_lisp_expr())
        self.feedback_requests.append(trees)
        self.encoded_requests.append(requested_values["encoding"])
        self.feedback_request_time = time.time()
        return trees

    def provide_feedback(self, feedback: int) -> bool:
        if not self.optimization_thread.is_alive():
            self.flush()
            return False
        if len(self.feedback_requests) == len(self.feedback_responses):
            return True
        self.feedback_duration.append(time.time() - self.feedback_request_time)
        self.feedback_responses.append(feedback)
        self.interpretability_estimate_updater.provide_feedback(
            encoded_trees=self.encoded_requests[self.feedback_counter],
            feedback=feedback
        )
        return True

    def flush(self) -> None:
        self.optimization_thread.join()
        tree_1, tree_2 = zip(*self.feedback_requests)
        feedback_data = pd.DataFrame(list(zip(
            self.feedback_duration, tree_1, tree_2, self.encoded_requests, self.feedback_responses)),
            columns=['duration', 'tree_1', 'tree_2', 'encoding', 'feedback'])
        feedback_data.to_csv(path_or_buf=path + "feedback-" + self.run_id + ".csv")
