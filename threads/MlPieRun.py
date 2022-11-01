import re
import time

import torch
from sympy import latex, parse_expr

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from genepro.node import Node
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from threads import OptimizationThread

import numpy as np

import pandas as pd


class MlPieRun:
    def __init__(self, run_id: str, optimization_thread: OptimizationThread,
                 interpretability_estimate_updater: InterpretabilityEstimateUpdater,
                 parameters: dict = None, path: str = None, ground_truth_computer: GroundTruthComputer = None):
        self.timeout_time = 3 * 60
        self.run_id = run_id
        self.optimization_thread: OptimizationThread = optimization_thread
        self.interpretability_estimate_updater: InterpretabilityEstimateUpdater = interpretability_estimate_updater
        self.path = path
        self.parameters = dict() if parameters is None else parameters
        self.feedback_counter: int = -1
        self.feedback_duration: list[float] = []
        self.feedback_requests: list[dict] = []
        self.encoded_requests: list[np.ndarray] = []
        self.feedback_predictions: list[int] = []
        self.feedback_responses: list[int] = []
        self.feedback_request_time: float = 0
        self.feedback_requests_iterations: list[int] = []
        self.feedback_responses_iterations: list[int] = []
        self.t1: Node = None
        self.t2: Node = None
        self.ground_truth_computer = ground_truth_computer

    def start(self) -> None:
        self.optimization_thread.start()

    def join(self) -> None:
        self.optimization_thread.join(5)

    def request_models(self) -> dict:
        if not self.optimization_thread.is_alive():
            self.flush()
            return {}
        # if the previous request was not answered I give it back again
        # might change it to -> sample again without incrementing the counter
        iteration = self.optimization_thread.get_current_iteration()
        total_generations = self.optimization_thread.termination[1]
        if len(self.feedback_requests) > len(self.feedback_responses):
            last_request = self.feedback_requests[self.feedback_counter]
            last_request['progress'] = 100 * iteration / total_generations
            return last_request

        requested_values = self.interpretability_estimate_updater.request_trees()
        if not requested_values:
            return {"wait": True}

        self.feedback_counter += 1
        self.feedback_requests_iterations.append(iteration)
        self.t1 = requested_values["t1"]
        self.t2 = requested_values["t2"]
        trees = (self.format_tree(self.t1), self.format_tree(self.t2))
        dictionary = {'models': list(trees), 'it': iteration, 'progress': 100 * iteration / total_generations}
        self.feedback_request_time = time.time()
        self.feedback_requests.append(dictionary)
        self.feedback_predictions.append(requested_values["prediction"])
        self.encoded_requests.append(requested_values["encoding"])
        return dictionary

    def request_progress(self) -> float:
        if not self.optimization_thread.is_alive():
            self.flush()
            return 100
        iteration = self.optimization_thread.get_current_iteration()
        total_generations = self.optimization_thread.termination[1]
        return 100 * iteration / total_generations

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
        self.optimization_thread.join(5)

        # prepare feedback file
        t1_latex, t1_parsable, t2_latex, t2_parsable = self.unwrap_requests(self.feedback_requests)
        feedback_data = pd.DataFrame(list(zip(
            self.feedback_duration, t1_latex, t1_parsable, t2_latex, t2_parsable, self.feedback_predictions,
            self.feedback_responses, self.feedback_requests_iterations, self.feedback_responses_iterations)),
            columns=['duration', 'tree_1_latex', 'tree_1_parsable', 'tree_2_latex', 'tree_2_parsable',
                     'prediction', 'feedback', 'req_iteration', 'resp_iteration'])
        # prepare nn file
        model = self.interpretability_estimate_updater.interpretability_estimator.get_net()

        # uncertainties
        uncertainties_df = {"generation": [], "average_uncertainty": [], "all_uncertainties": []}
        uncertainties = self.optimization_thread.problem.get_uncertainties()
        initial_avg_uncertainty = sum(uncertainties[0]) / len(uncertainties[0])
        for iii in range(len(uncertainties)):
            gen_uncertainties = uncertainties[iii]
            current_avg_uncertainty = sum(gen_uncertainties) / len(gen_uncertainties)
            current_avg_uncertainty = current_avg_uncertainty / initial_avg_uncertainty
            uncertainties_df["generation"].append(iii)
            uncertainties_df["average_uncertainty"].append(current_avg_uncertainty)
            sss = ""
            for ggg in gen_uncertainties:
                sss += str(ggg)
                sss += "|"
            sss = sss[:len(sss) - 1]
            uncertainties_df["all_uncertainties"].append(sss)
        uncertainties_df = pd.DataFrame(uncertainties_df)

        # write optimization file
        parsable_trees, latex_trees, accuracies, interpretabilities, ground_truth_values = self.parse_optimization_history(
            self.optimization_thread.result.opt, self.ground_truth_computer)
        best_data = pd.DataFrame(
            list(zip(parsable_trees, latex_trees, accuracies, interpretabilities, ground_truth_values)),
            columns=['parsable_tree', 'latex_tree', 'accuracy', 'interpretability', 'ground_truth_value'])

        # update dataframes
        for k in self.parameters.keys():
            feedback_data[k] = self.parameters[k]
            best_data[k] = self.parameters[k]

        best_data['rerun'] = 'false'

        # save files
        uncertainties_df.to_csv(path_or_buf=self.path + "uncertainty-" + self.run_id + ".csv")
        feedback_data.to_csv(path_or_buf=self.path + "feedback-" + self.run_id + ".csv")
        torch.save(model, self.path + "nn-" + self.run_id + ".pth")
        best_data.to_csv(path_or_buf=self.path + "best-" + self.run_id + ".csv")

    def is_abandoned(self) -> bool:
        return time.time() - self.feedback_request_time > self.timeout_time

    def get_pareto_front(self) -> pd.DataFrame:
        if self.optimization_thread.is_alive():
            return None
        front = self.optimization_thread.result.opt
        parsable_trees, latex_trees, accuracies, interpretabilities, ground_truth_values = self.parse_front(front, self.ground_truth_computer)
        return pd.DataFrame(list(zip(accuracies, interpretabilities, ground_truth_values, parsable_trees, latex_trees)),
                            columns=['accuracy', 'interpretability', 'ground_truth_value', 'parsable_tree', 'latex_tree'])

    @staticmethod
    def safe_latex_format(tree: Node) -> str:
        readable_repr = tree.get_readable_repr().replace("u-", "-")
        try:
            latex_repr = latex(parse_expr(readable_repr, evaluate=False))
        except (RuntimeError, TypeError, ZeroDivisionError, Exception) as e:
            latex_repr = readable_repr
        return re.sub(r"(\.[0-9][0-9])(\d+)", r"\1", latex_repr)

    @staticmethod
    def format_tree(tree: Node) -> dict:
        latex_repr = MlPieRun.safe_latex_format(tree)
        parsable_repr = str(tree.get_subtree())
        return {"latex": latex_repr, "parsable": parsable_repr}

    @staticmethod
    def unwrap_requests(list_of_requests: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
        t1_latex = []
        t1_parsable = []
        t2_latex = []
        t2_parsable = []
        for a_dict in list_of_requests:
            models = a_dict['models']
            t1_latex.append(models[0]["latex"])
            t2_latex.append(models[1]["latex"])
            t1_parsable.append(models[0]["parsable"])
            t2_parsable.append(models[1]["parsable"])
        return t1_latex, t1_parsable, t2_latex, t2_parsable

    @staticmethod
    def parse_optimization_history(optimal, ground_truth_computer: GroundTruthComputer = None) -> tuple[list[str], list[str], list[float], list[float], list[float]]:
        parsable_trees = []
        latex_trees = []
        accuracies = []
        interpretabilities = []
        ground_truth_values = []

        for individual in optimal:
            tree = individual.X[0]
            parsable_trees.append(str(tree.get_subtree()))
            latex_trees.append(tree.get_readable_repr().replace("u-", "-"))
            accuracies.append(individual.F[0])
            interpretabilities.append(individual.F[1])
            if ground_truth_computer is not None:
                ground_truth_values.append(ground_truth_computer.compute(tree))
            else:
                ground_truth_values.append(1234.0)

        return parsable_trees, latex_trees, accuracies, interpretabilities, ground_truth_values

    @staticmethod
    def parse_front(optimal, ground_truth_computer: GroundTruthComputer = None) -> tuple[list[str], list[str], list[float], list[float], list[float]]:
        parsable_trees = []
        latex_trees = []
        accuracies = []
        interpretabilities = []
        ground_truth_values = []

        for individual in optimal:
            tree = individual.X[0]
            parsable_trees.append(str(tree.get_subtree()))
            latex_trees.append(MlPieRun.safe_latex_format(tree))
            accuracies.append(individual.F[0])
            interpretabilities.append(individual.F[1])
            if ground_truth_computer is not None:
                ground_truth_values.append(ground_truth_computer.compute(tree))
            else:
                ground_truth_values.append(1234.0)

        return parsable_trees, latex_trees, accuracies, interpretabilities, ground_truth_values
