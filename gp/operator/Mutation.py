from abc import abstractmethod, ABC
from gp.tree.PrimitiveTree import PrimitiveTree


class Mutation(ABC):

    @abstractmethod
    def mutate(self, individual: PrimitiveTree) -> PrimitiveTree:
        pass
