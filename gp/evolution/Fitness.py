from abc import abstractmethod, ABC
from typing import List
from gp.evolution.Population import Population
from gp.tree.PrimitiveTree import PrimitiveTree


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
    def evaluate(self, individual: PrimitiveTree) -> List[float]:
        pass
