from abc import abstractmethod, ABC
from typing import List, Tuple

from genepro.node import Node


class FeedbackCollector(ABC):

    @abstractmethod
    def collect_feedback(self, pairs: List[Tuple[Node, Node]]) -> List[int]:
        pass
