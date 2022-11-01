import threading
import time
from typing import Tuple, List, Dict, Any
import random
import torch
import numpy as np
import warnings

from numpy import VisibleDeprecationWarning
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem

from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from copy import deepcopy

from nsgp.evaluation.TreeEvaluator import TreeEvaluator
from nsgp.evolution.NSGP2 import NSGP2
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.MultiObjectiveMinimizationElementWiseProblem import MultiObjectiveMinimizationElementWiseProblem
from nsgp.problem.MultiObjectiveMinimizationProblem import MultiObjectiveMinimizationProblem

from nsgp.structure.TreeStructure import TreeStructure


class GPWithNSGA2:
    def __init__(self, structure: TreeStructure,
                 evaluators: List[TreeEvaluator],
                 pop_size: int,
                 num_gen: int,
                 element_wise_eval: bool = False,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.6,
                 num_offsprings: int = None,
                 duplicates_elimination_data: np.ndarray = None,
                 callback: Callback = None
                 ):
        self.__structure = structure
        self.__element_wise_eval = element_wise_eval
        self.__evaluators = deepcopy(evaluators)
        self.__number_of_evaluators = len(self.__evaluators)
        self.__pop_size = pop_size
        self.__num_gen = num_gen
        self.__num_offsprings = num_offsprings
        if callback is None:
            self.__callback = Callback()
        else:
            self.__callback = callback
        if self.__num_offsprings is None:
            self.__num_offsprings = self.__pop_size
        self.__tournament_selection = TournamentSelection(func_comp=binary_tournament, pressure=2)
        self.__duplicates_elimination_data = duplicates_elimination_data
        self.__setting = TreeSetting(self.__structure, self.__duplicates_elimination_data, crossover_prob=crossover_prob, mutation_prob=mutation_prob)
        self.__tree_sampling = self.__setting.get_sampling()
        self.__tree_crossover = self.__setting.get_crossover()
        self.__tree_mutation = self.__setting.get_mutation()
        self.__duplicates_elimination = self.__setting.get_duplicates_elimination()
        self.__algorithm = NSGP2(pop_size=self.__pop_size,
                                 n_offsprings=self.__num_offsprings,
                                 selection=self.__tournament_selection,
                                 sampling=self.__tree_sampling,
                                 crossover=self.__tree_crossover,
                                 mutation=self.__tree_mutation,
                                 eliminate_duplicates=self.__duplicates_elimination
                                 )

    def run_minimization(self, seed: int = None, verbose: bool = False, save_history: bool = False, mutex: threading.Lock = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        if self.__element_wise_eval:
            problem: Problem = MultiObjectiveMinimizationElementWiseProblem(self.__evaluators, mutex)
        else:
            problem: Problem = MultiObjectiveMinimizationProblem(self.__evaluators, mutex)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=VisibleDeprecationWarning)
            res = minimize(problem=problem, algorithm=deepcopy(self.__algorithm),
                           termination=("n_gen", self.__num_gen),
                           seed=seed, verbose=verbose, save_history=save_history,
                           callback=deepcopy(self.__callback),
                           return_least_infeasible=False)
        end = time.time()
        if verbose:
            print("")
            print(f"Run with seed {seed} completed!")
            print("")
        return {"result": res, "executionTimeInHours": (end - start)*(1/3600), "seed": seed}
