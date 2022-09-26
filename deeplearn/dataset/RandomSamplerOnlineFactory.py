from typing import List
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.dataset.PairSamplerFactory import PairSamplerFactory
from deeplearn.dataset.RandomSamplerOnline import RandomSamplerOnline


class RandomSamplerOnlineFactory(PairSamplerFactory):

    def create_sampler(self, n_pairs: int = 20, already_seen: List[int] = None) -> PairSampler:
        return RandomSamplerOnline(n_pairs, already_seen)
