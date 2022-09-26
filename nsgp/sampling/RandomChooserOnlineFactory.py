from typing import Set

from genepro.node import Node
from nsgp.sampling.PairChooser import PairChooser

from nsgp.sampling.PairChooserFactory import PairChooserFactory
from nsgp.sampling.RandomChooserOnline import RandomChooserOnline


class RandomChooserOnlineFactory(PairChooserFactory):

    def create(self, n_pairs: int = 1, already_seen: Set[Node] = None) -> PairChooser:
        return RandomChooserOnline(n_pairs, already_seen)
