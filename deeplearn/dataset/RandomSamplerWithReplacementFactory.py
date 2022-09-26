from typing import List
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.dataset.PairSamplerFactory import PairSamplerFactory
from deeplearn.dataset.RandomSamplerWithReplacement import RandomSamplerWithReplacement


class RandomSamplerWithReplacementFactory(PairSamplerFactory):

    def create_sampler(self, n_pairs: int = 20, already_seen: List[int] = None) -> PairSampler:
        return RandomSamplerWithReplacement(n_pairs, already_seen)
