import random

from gp.tree.FullGenerator import FullGenerator
from gp.tree.GrowGenerator import GrowGenerator
from gp.tree.PrimitiveSet import PrimitiveSet
from gp.tree.PrimitiveTree import PrimitiveTree
from gp.tree.TerminalSet import TerminalSet
from gp.tree.TreeGenerator import TreeGenerator


class HalfHalfGenerator(TreeGenerator):
    def __init__(self, primitive_set: PrimitiveSet, terminal_set: TerminalSet, min_height: int, max_height: int):
        self.__primitive_set = primitive_set
        self.__terminal_set = terminal_set
        self.__min_height = min_height
        self.__max_height = max_height

    def generate_tree(self) -> PrimitiveTree:
        ind = random.randint(0, 1)
        if ind == 0:
            return FullGenerator(self.__primitive_set, self.__terminal_set, self.__min_height, self.__max_height).generate_tree()
        else:
            return GrowGenerator(self.__primitive_set, self.__terminal_set, self.__min_height, self.__max_height).generate_tree()
