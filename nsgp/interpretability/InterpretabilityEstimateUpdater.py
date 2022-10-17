import threading
from copy import deepcopy

import numpy as np
import torch

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.trainer.Trainer import Trainer
from deeplearn.trainer.blockingtrainer.BlockingBatchTrainer import BlockingBatchTrainer
from deeplearn.trainer.blockingtrainer.BlockingOnlineTrainer import BlockingOnlineTrainer
from deeplearn.trainer.blockingtrainer.BlockingTrainer import BlockingTrainer
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.sampling.PairChooserFactory import PairChooserFactory


class InterpretabilityEstimateUpdater:
    def __init__(self, individuals: set, mutex: threading.Lock, interpretability_estimator: Trainer,
                 encoder: TreeEncoder, pair_chooser: PairChooserFactory, batch_size: int = 1) -> None:
        self.nn_updater: BlockingTrainer = BlockingOnlineTrainer() if batch_size == 1 else BlockingBatchTrainer(
            batch_size)
        self.pair_chooser = pair_chooser.create(1)
        self.individuals = individuals
        self.mutex = mutex
        self.interpretability_estimator = interpretability_estimator
        self.encoder = encoder

    def request_trees(self) -> dict:
        if len(self.individuals) < 2:
            return {}
        local_individuals = deepcopy(self.individuals)
        t1, t2 = self.pair_chooser.sample(local_individuals, self.encoder, self.interpretability_estimator, self.mutex)[
            0]
        t1_encoded = self.encoder.encode(t1, True)
        t2_encoded = self.encoder.encode(t2, True)
        encoded_trees = np.concatenate((t1_encoded, t2_encoded), axis=None).reshape(1, -1)
        with self.mutex:
            i1_prediction = \
                self.interpretability_estimator.predict(torch.from_numpy(t1_encoded).float().reshape(1, -1))[0][0][0].item()
            i2_prediction = \
                self.interpretability_estimator.predict(torch.from_numpy(t2_encoded).float().reshape(1, -1))[0][0][0].item()
        prediction = -1 if i1_prediction >= i2_prediction else 1
        return {
            "t1": t1,
            "t2": t2,
            "encoding": encoded_trees,
            "prediction": prediction
        }

    def provide_feedback(self, encoded_trees, feedback):
        data = NumericalData(encoded_trees, np.array([feedback]))
        self.nn_updater.update(self.interpretability_estimator, data, self.mutex)
