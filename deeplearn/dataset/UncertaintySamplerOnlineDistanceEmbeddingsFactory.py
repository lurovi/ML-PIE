from typing import List
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.dataset.PairSamplerFactory import PairSamplerFactory
from deeplearn.dataset.UncertaintySamplerOnlineDistanceEmbeddings import UncertaintySamplerOnlineDistanceEmbeddings


class UncertaintySamplerOnlineDistanceEmbeddingsFactory(PairSamplerFactory):
    def __init__(self, normalization_func: str = "max", lambda_coeff: float = 0.5):
        super().__init__()
        self.__lambda_coeff = lambda_coeff
        self.__normalization_func_name = normalization_func

    def create_sampler(self, n_pairs: int = 20, already_seen: List[int] = None) -> PairSampler:
        return UncertaintySamplerOnlineDistanceEmbeddings(n_pairs, already_seen, self.__normalization_func_name, self.__lambda_coeff)
