from typing import Set

from genepro.node import Node
from nsgp.sampling.PairChooser import PairChooser

from nsgp.sampling.PairChooserFactory import PairChooserFactory
from nsgp.sampling.UncertaintyChooserDistanceEmbeddings import UncertaintyChooserDistanceEmbeddings


class UncertaintyChooserDistanceEmbeddingsFactory(PairChooserFactory):
    def __init__(self, normalization_func: str = "max", lambda_coeff: float = 0.5):
        super().__init__()
        self.__normalization_func_name = normalization_func
        self.__lambda_coeff = lambda_coeff

    def create(self, n_pairs: int = 1, already_seen: Set[Node] = None) -> PairChooser:
        return UncertaintyChooserDistanceEmbeddings(n_pairs, already_seen, self.__normalization_func_name, self.__lambda_coeff)
