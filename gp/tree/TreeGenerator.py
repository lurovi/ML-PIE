from abc import abstractmethod, ABC
from gp.tree.PrimitiveTree import PrimitiveTree


class TreeGenerator(ABC):

    @abstractmethod
    def generate_tree(self) -> PrimitiveTree:
        pass

    @staticmethod
    def gen_simple_leaf_tree_as_list(leaf: str, max_degree: int, max_depth: int) -> List[List[str]]:
        tre = [[leaf]]
        for i in range(1, max_depth):
            tre.append([""] * (max_degree ** i))
        return tre
