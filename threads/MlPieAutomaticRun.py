import time

import numpy as np

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from nsgp.sampling.FeedbackCollector import FeedbackCollector
from threads.MlPieRun import MlPieRun
from threads.OptimizationThread import OptimizationThread


class MlPieAutomaticRun(MlPieRun):
    def __init__(self, run_id: str, optimization_thread: OptimizationThread,
                 interpretability_estimate_updater: InterpretabilityEstimateUpdater,
                 feedback_collector: FeedbackCollector, parameters: dict = None,
                 path: str = None, ground_truth_computer: GroundTruthComputer = None):
        super().__init__(run_id, optimization_thread, interpretability_estimate_updater, parameters, path, ground_truth_computer=ground_truth_computer)
        self.feedback_collector: FeedbackCollector = feedback_collector

    def run_automatically(self, delay: float, method_sleep: str = "random_uniform"):
        half_num_gen = self.optimization_thread.termination[1] // 2
        self.start()
        time.sleep(5)
        self.optimization_thread.callback.population_non_empty.wait()
        while True:
            dictionary = self.request_models()
            if not dictionary:
                break
            feedback = self.feedback_collector.collect_feedback([(self.t1, self.t2)])[0]
            curr_iter = self.optimization_thread.get_current_iteration()
            curr_iter = 1 if curr_iter is None else curr_iter

            if method_sleep == "random_uniform":
                time.sleep(np.random.uniform(1, delay + 1e-4))
            elif method_sleep == "constant_rate":
                time.sleep(delay)
            elif method_sleep == "lazy_start":
                start_time, end_time = 2 * delay - 2, 2
                if curr_iter <= half_num_gen:
                    time.sleep(start_time)
                else:
                    time.sleep(end_time)
            elif method_sleep == "lazy_end":
                start_time, end_time = 2, 2 * delay - 2
                if curr_iter <= half_num_gen:
                    time.sleep(start_time)
                else:
                    time.sleep(end_time)

            if not self.provide_feedback(feedback):
                break
        self.join()
