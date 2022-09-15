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

    def notify(self, algorithm):
        for p in algorithm.pop:
            self.population_storage.add(deepcopy(p.X[0]))
        self.population_non_empty.set()
