import threading
from copy import deepcopy
import time

import numpy as np

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.trainer.Trainer import Trainer
from deeplearn.trainer.blockingtrainer.BlockingBatchTrainer import BlockingBatchTrainer
from deeplearn.trainer.blockingtrainer.BlockingOnlineTrainer import BlockingOnlineTrainer
from deeplearn.trainer.blockingtrainer.BlockingTrainer import BlockingTrainer
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.sampling.FeedbackCollector import FeedbackCollector
from nsgp.sampling.PairChooser import PairChooser


class InterpretabilityEstimateUpdater:
    def __init__(self, individuals: set, mutex: threading.Lock, interpretability_estimator: Trainer,
                 encoder: TreeEncoder, pair_chooser: PairChooser, feedback_collector: FeedbackCollector,
                 batch_size: int = 1) -> None:
        self.nn_updater: BlockingTrainer = BlockingOnlineTrainer() if batch_size == 1 else BlockingBatchTrainer(
            batch_size)
        self.feedback_collector = feedback_collector
        self.pair_chooser = pair_chooser
        self.individuals = individuals
        self.mutex = mutex
        self.interpretability_estimator = interpretability_estimator
        self.encoder = encoder

    def immediate_update(self):
        if len(self.individuals) < 2:
            return
        local_individuals = deepcopy(self.individuals)
        selected = self.pair_chooser.sample(local_individuals, self.encoder, self.interpretability_estimator)
        feedback = self.feedback_collector.collect_feedback(selected)[0]
        t1_encoded = self.encoder.encode(selected[0][0], True)
        t2_encoded = self.encoder.encode(selected[0][1], True)
        encoded_trees = np.concatenate((t1_encoded, t2_encoded), axis=None).reshape(1, -1)
        data = NumericalData(encoded_trees, np.array([feedback]))
        self.nn_updater.update(self.interpretability_estimator, data, self.mutex)

    def delayed_update(self, delay: float = 0.5):
        time.sleep(delay)
        self.immediate_update()
