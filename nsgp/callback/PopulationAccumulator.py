import threading
from copy import deepcopy

from pymoo.core.callback import Callback


class PopulationAccumulator(Callback):

    def __init__(self, population_storage: set = None) -> None:
        super().__init__()
        if population_storage is None:
            population_storage = set()
        self.population_storage = population_storage
        self.population_non_empty = threading.Event()
        self.__iterations_counter = -1

    def notify(self, algorithm):
        old_population = set()
        for old_individual in self.population_storage:
            old_population.add(old_individual)
        for p in algorithm.pop:
            individual = deepcopy(p.X[0])
            old_population.discard(individual)
            self.population_storage.add(individual)
        for old_individual in old_population:
            self.population_storage.discard(old_individual)

        self.population_non_empty.set()
        self.__iterations_counter += 1

    def get_current_iteration(self) -> int:
        return self.__iterations_counter
