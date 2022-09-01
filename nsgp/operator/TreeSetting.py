from nsgp.operator.DuplicateTreeElimination import DuplicateTreeElimination
from nsgp.operator.TreeCrossover import TreeCrossover
from nsgp.operator.TreeMutation import TreeMutation
from nsgp.operator.TreeSampling import TreeSampling
from util.TreeGrammarStructure import TreeGrammarStructure
import numpy as np


class TreeSetting:
    def __init__(self, structure: TreeGrammarStructure, little_data: np.ndarray):
        self.__structure: TreeGrammarStructure = structure
        self.__little_data: np.ndarray = little_data
        self.__sampling: TreeSampling = TreeSampling(structure)
        self.__crossover: TreeCrossover = TreeCrossover(structure)
        self.__mutation: TreeMutation = TreeMutation(structure)
        self.__duplicates_elimination: DuplicateTreeElimination = DuplicateTreeElimination(little_data)

    def get_sampling(self) -> TreeSampling:
        return self.__sampling

    def get_crossover(self) -> TreeCrossover:
        return self.__crossover

    def get_mutation(self) -> TreeMutation:
        return self.__mutation

    def get_duplicates_elimination(self) -> DuplicateTreeElimination:
        return self.__duplicates_elimination
