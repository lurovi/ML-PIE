from copy import deepcopy
import random
import time

import numpy as np

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.trainer.Trainer import Trainer
from deeplearn.trainer.blockingtrainer.BlockingBatchTrainer import BlockingBatchTrainer
from deeplearn.trainer.blockingtrainer.BlockingOnlineTrainer import BlockingOnlineTrainer
from deeplearn.trainer.blockingtrainer.BlockingTrainer import BlockingTrainer
from nsgp.encoder.TreeEncoder import TreeEncoder


class InterpretabilityEstimateUpdater:
    def __init__(self, individuals: set, mutex, interpretability_estimator: Trainer, encoder: TreeEncoder,
                 candidate_selector=None, feedback_collector=None, batch_size: int = 1) -> None:
        super().__init__()
        self.nn_updater: BlockingTrainer = BlockingOnlineTrainer() if batch_size == 1 else BlockingBatchTrainer(
            batch_size)
        self.feedback_provider = feedback_collector
        self.candidate_selector = candidate_selector
        self.individuals = individuals
        self.mutex = mutex
        self.interpretability_estimator = interpretability_estimator
        self.encoder = encoder

    def collect_feedback(self):
        local_individuals = deepcopy(self.individuals)

        # t1, t2 = self.candidate_selector.select(local_individuals, self.interpretability_estimator, self.encoder)
        # feedback = self.feedback_provider.evaluate(t1, t2)

        # dummy workaround
        selected = random.sample(local_individuals, 2)
        t1 = selected[0]
        t2 = selected[1]
        feedback = random.random()
        time.sleep(0.5)
        print("FEED")

        t1_encoded = self.encoder.encode(t1, True)
        t2_encoded = self.encoder.encode(t2, True)
        encoded_trees = np.concatenate((t1_encoded, t2_encoded), axis=None).reshape(1, -1)
        data = NumericalData(encoded_trees, np.array([feedback]))
        self.nn_updater.update(self.interpretability_estimator, data, self.mutex)
