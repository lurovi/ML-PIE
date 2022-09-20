import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import binary_tournament, NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize

from nsgp.operator.TreeSetting import TreeSetting

from nsgp.structure.TreeStructure import TreeStructure


class GPWithNSGA2:
    def __init__(self, structure: TreeStructure,
                 problem: Problem,
                 pop_size: int,
                 num_gen: int,
                 tournament_size: int = 5,
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
        self.__callback = callback
        if self.__num_offsprings is None:
            self.__num_offsprings = self.__pop_size
        self.__tournament_size = tournament_size
        self.__tournament_selection = TournamentSelection(func_comp=binary_tournament, pressure=self.__tournament_size)
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

    def run_minimization(self, seed: int = None, verbose: bool = True, save_history: bool = True) -> Result:
        start = time.time()
        res = minimize(problem=self.__problem, algorithm=self.__algorithm, termination=("n_gen", self.__num_gen),
                        seed=seed, verbose=verbose, save_history=save_history, callback=self.__callback)
        end = time.time()
        return res, (end - start)*(1/3600)
