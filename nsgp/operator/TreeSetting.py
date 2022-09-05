from nsgp.operator.DuplicateTreeElimination import DuplicateTreeElimination
from nsgp.operator.TreeCrossover import TreeCrossover
from nsgp.operator.TreeMutation import TreeMutation
from nsgp.operator.TreeSampling import TreeSampling
from nsgp.util.TreeGrammarStructure import TreeGrammarStructure
import numpy as np


class TreeSetting:
    def __init__(self, structure: TreeGrammarStructure, little_data: np.ndarray):
        if little_data.shape[1] != structure.get_number_of_features():
            raise AttributeError(f"The number of features declared is {structure.get_number_of_features()}. However, little_data parameter of TreeSetting constructor is filled with a numpy matrix whose number of columns is different: {little_data.shape[1]}.")
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
