from typing import List
from gp.evolution.Fitness import Fitness
from gp.tree.PrimitiveTree import PrimitiveTree
import numpy as np


class MSESizeFitness(Fitness):
    def __init__(self, X, y, weights: List[float] = []):
        super(MSESizeFitness, self).__init__(weights)
        self.X = X if isinstance(X, np.ndarray) else np.array(X)
        self.y = y if isinstance(y, np.ndarray) else np.array(y)

    def evaluate(self, individual: PrimitiveTree) -> List[float]:
        pred = np.zeros(self.y.shape[0])
        num_nodes = individual.number_of_nodes()
        for i in range(len(self.X)):
            pred[i] = individual.compile(self.X[i])
        return [np.sum(np.square(pred - self.y))/float(self.X.shape[0]), -1.0/float(num_nodes)]
