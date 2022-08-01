import numpy as np
from typing import List
from gp.tree.PrimitiveTree import PrimitiveTree


class Population:
    def __init__(self, population: List[PrimitiveTree]):
        self.__size = len(population)
        self.__population = population
        self.__fitness = [[] for _ in range(self.__size)]

    def has_fitness(self, idx: int) -> bool:
        if not (0 <= idx < self.get_size()):
            raise IndexError(f"{idx} is out of range as index for this population.")
        return len(self.__fitness[idx]) != 0

    def get_size(self) -> int:
        return self.__size

    def get_fitness(self, idx: int) -> List[float]:
        if not (0 <= idx < self.get_size()):
            raise IndexError(f"{idx} is out of range as index for this population.")
        return self.__fitness[idx]

    def get_individual(self, idx: int) -> PrimitiveTree:
        if not (0 <= idx < self.get_size()):
            raise IndexError(f"{idx} is out of range as index for this population.")
        return self.__population[idx]

    def statistics(self):
        v = np.array(self.__fitness)
        mean = np.mean(v, axis=0)
        std = np.std(v, axis=0)
        max_ = np.max(v, axis=0)
        min_ = np.min(v, axis=0)
        return {"MEAN": mean, "STD_DEV": std, "MAX": max_, "MIN": min_}

    def update_population(self, population: List[PrimitiveTree], fitness: List[List[float]]):
        if len(population) != len(fitness):
            raise AttributeError("The size of the population must be equal to the size of the fitness array.")
        self.__size = len(population)
        self.__population = population
        self.__fitness = fitness
