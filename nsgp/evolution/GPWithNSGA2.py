import time
from typing import Tuple
import random
import torch
import numpy as np
from pymoo.algorithms.moo.nsga2 import binary_tournament, NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from copy import deepcopy

from pymoo.termination import get_termination

from nsgp.operator.TreeSetting import TreeSetting

from nsgp.structure.TreeStructure import TreeStructure


class GPWithNSGA2:
    def __init__(self, structure: TreeStructure,
                 problem: Problem,
                 pop_size: int,
                 num_gen: int,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.6,
                 num_offsprings: int = None,
                 duplicates_elimination_data: np.ndarray = None,
                 callback: Callback = None):
        self.__structure = structure
        self.__problem = problem
        self.__pop_size = pop_size
        self.__num_gen = num_gen
        self.__num_offsprings = num_offsprings
        if callback is None:
            self.__callback = Callback()
        else:
            self.__callback = callback
        self.__termination = get_termination("n_gen", self.__num_gen)
        if self.__num_offsprings is None:
            self.__num_offsprings = self.__pop_size
        self.__tournament_selection = TournamentSelection(func_comp=binary_tournament, pressure=2)
        self.__duplicates_elimination_data = duplicates_elimination_data
        self.__setting = TreeSetting(self.__structure, self.__duplicates_elimination_data, crossover_prob=crossover_prob, mutation_prob=mutation_prob)
        self.__tree_sampling = self.__setting.get_sampling()
        self.__tree_crossover = self.__setting.get_crossover()
        self.__tree_mutation = self.__setting.get_mutation()
        self.__duplicates_elimination = self.__setting.get_duplicates_elimination()
        self.__algorithm = NSGA2(pop_size=self.__pop_size,
                                 n_offsprings=self.__num_offsprings,
                                 selection=self.__tournament_selection,
                                 sampling=self.__tree_sampling,
                                 crossover=self.__tree_crossover,
                                 mutation=self.__tree_mutation,
                                 eliminate_duplicates=self.__duplicates_elimination
                                 )

    def run_minimization(self, seed: int = None, verbose: bool = True, save_history: bool = True) -> Tuple[Result, float]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        start = time.time()
        res = minimize(problem=deepcopy(self.__problem), algorithm=deepcopy(self.__algorithm),
                       termination=deepcopy(self.__termination),
                       seed=seed, verbose=verbose, save_history=save_history,
                       callback=deepcopy(self.__callback),
                       return_least_infeasible=False)
        end = time.time()
        return res, (end - start)*(1/3600)
