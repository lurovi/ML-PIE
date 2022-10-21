import threading
from copy import deepcopy
from typing import Any

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.problem import Problem

from pymoo.optimize import minimize

from nsgp.callback.PopulationAccumulator import PopulationAccumulator


class OptimizationThread(threading.Thread):

    def __init__(self, optimization_algorithm: GeneticAlgorithm, problem: Problem, termination: Any, seed: int,
                 callback: PopulationAccumulator = None, verbose: bool = False, save_history: bool = False):
        threading.Thread.__init__(self)
        self.optimization_algorithm = optimization_algorithm
        self.problem = problem
        self.termination = termination
        self.seed = seed
        self.callback = callback
        self.verbose = verbose
        self.save_history = save_history
        self.result = None

    def run(self) -> None:
        self.result = minimize(
            problem=self.problem,
            algorithm=self.optimization_algorithm,
            termination=self.termination,
            seed=self.seed,
            verbose=self.verbose,
            callback=self.callback,
            save_history=self.save_history
        )

    def get_current_iteration(self) -> int:
        if self.callback is not None:
            return self.callback.get_current_iteration()

    def get_result(self):
        return self.result
