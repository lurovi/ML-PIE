from abc import abstractmethod, ABC
from typing import List
from deeplearn.dataset.PairSampler import PairSampler


class PairSamplerFactory(ABC):

    @abstractmethod
    def create_sampler(self, n_pairs: int = 20, already_seen: List[int] = None) -> PairSampler:
        pass
