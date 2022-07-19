from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from gp.tree import *


# ==============================================================================================================
# INDIVIDUAL AND POPULATION
# ==============================================================================================================


class Individual(ABC):

    @abstractmethod
    def get_individual(self):
        pass


class PrimitiveTreeIndividual(Individual):
    def __int__(self, individual: PrimitiveTree):
        self.__individual = individual

    def get_individual(self):
        return self.__individual


class Population:
    def __init__(self, population: List[Individual]):
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

    def get_individual(self, idx: int) -> Individual:
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

    def update_population(self, population: List[Individual], fitness: List[List[float]]):
        if len(population) != len(fitness):
            raise AttributeError("The size of the population must be equal to the size of the fitness array.")
        self.__size = len(population)
        self.__population = population
        self.__fitness = fitness


# ==============================================================================================================
# FITNESS
# ==============================================================================================================


class Fitness(ABC):
    def __init__(self, names: List[str], weights: List[float] = []):
        self.__names = names
        self.__weights = weights
        if self.__weights and not( not([nnn for nnn in self.__weights if nnn < 0.0]) and sum(self.__weights) == 1.0 ) and not( len(self.__weights) == len(self.__names) ):
            raise AttributeError("If you specify a weights array in the Fitness class constructor, then the sum of the elements in the array must be 1 and each weight must be greater than or equal to 0. Moreover, the length of the weights array must be equal to the length of the names array, i.e., there must be a weight in the weights array for each different objective function.")

    def get_weights(self):
        return self.__weights

    def get_names(self):
        return self.__names

    def statistics(self, population: Population):
        stats = population.statistics()
        names_stats = {}
        for key in stats.keys():
            names_stats[key] = {}
            curr_stat = stats[key]
            t = 0
            for name in self.get_names():
                names_stats[key][name] = curr_stat[t]
                t += 1
        return names_stats

    def aggregate(self, fitness_eval: List[float]):
        if len(fitness_eval) != len(self.__weights):
            raise AttributeError("The length of weights array must be equal to the length of the provided fitness values array.")
        return sum([self.__weights[i]*fitness_eval[i] for i in range(len(fitness_eval))])

    @abstractmethod
    def evaluate(self, individual: Individual) -> List[float]:
        pass


class MSESizeFitness(Fitness):
    def __init__(self, X, y, weights: List[float] = []):
        super(MSESizeFitness, self).__init__(weights)
        self.X = X if isinstance(X, np.ndarray) else np.array(X)
        self.y = y if isinstance(y, np.ndarray) else np.array(y)

    def evaluate(self, individual: Individual) -> List[float]:
        pred = np.zeros(self.y.shape[0])
        curr_individual = individual.get_individual()
        num_nodes = curr_individual.number_of_nodes()
        for i in range(len(self.X)):
            pred[i] = curr_individual.compile(self.X[i])
        return [np.sum(np.square(pred - self.y))/float(self.X.shape[0]), -1.0/float(num_nodes)]






