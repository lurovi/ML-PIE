from nsgp.operator.DuplicateTreeElimination import DuplicateTreeElimination
from nsgp.operator.TreeCrossover import TreeCrossover
from nsgp.operator.TreeMutation import TreeMutation
from nsgp.operator.TreeSampling import TreeSampling
from nsgp.structure.TreeStructure import TreeStructure
import numpy as np


class TreeSetting:
    def __init__(self, structure: TreeStructure, little_data: np.ndarray = None, crossover_prob: float = 0.9, mutation_prob: float = 0.6):
        if little_data is not None and little_data.shape[1] != structure.get_number_of_features():
            raise AttributeError(f"The number of features declared is {structure.get_number_of_features()}. However, little_data parameter of TreeSetting constructor is filled with a numpy matrix whose number of columns is different: {little_data.shape[1]}.")
        self.__structure: TreeStructure = structure
        self.__little_data: np.ndarray = little_data
        self.__sampling: TreeSampling = TreeSampling(structure)
        self.__crossover: TreeCrossover = TreeCrossover(structure, prob=crossover_prob)
        self.__mutation: TreeMutation = TreeMutation(structure, prob=mutation_prob)
        self.__duplicates_elimination: DuplicateTreeElimination = DuplicateTreeElimination(little_data)

    def get_sampling(self) -> TreeSampling:
        return self.__sampling

    def get_crossover(self) -> TreeCrossover:
        return self.__crossover

    def get_mutation(self) -> TreeMutation:
        return self.__mutation

    def get_duplicates_elimination(self) -> DuplicateTreeElimination:
        return self.__duplicates_elimination
