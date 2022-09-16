from typing import List
from deeplearn.dataset.PairSampler import PairSampler
from deeplearn.dataset.PairSamplerFactory import PairSamplerFactory
from deeplearn.dataset.UncertaintySamplerOnline import UncertaintySamplerOnline


class UncertaintySamplerOnlineFactory(PairSamplerFactory):

    def create_sampler(self, n_pairs: int = 20, already_seen: List[int] = None) -> PairSampler:
        return UncertaintySamplerOnline(n_pairs, already_seen)
