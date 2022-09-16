from typing import List
from deeplearn.dataset.PairSamplerFactory import PairSamplerFactory
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.dataset.RandomSampler import RandomSampler


class RandomSamplerFactory(PairSamplerFactory):

    def create_sampler(self, n_pairs: int = 20, already_seen: List[int] = None) -> PairSampler:
        return RandomSampler(n_pairs, already_seen)
