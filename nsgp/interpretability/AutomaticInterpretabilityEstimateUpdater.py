import threading

import time

from deeplearn.trainer.Trainer import Trainer
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from nsgp.sampling.FeedbackCollector import FeedbackCollector
from nsgp.sampling.PairChooser import PairChooser


class AutomaticInterpretabilityEstimateUpdater(InterpretabilityEstimateUpdater):
    def __init__(self, individuals: set, mutex: threading.Lock, interpretability_estimator: Trainer,
                 encoder: TreeEncoder, pair_chooser: PairChooser, feedback_collector: FeedbackCollector,
                 batch_size: int = 1) -> None:
        super().__init__(individuals, mutex, interpretability_estimator, encoder, pair_chooser, batch_size)
        self.feedback_collector = feedback_collector

    def immediate_update(self):
        request_result = self.request_trees()
        if not request_result:
            return
        t1 = request_result["t1"]
        t2 = request_result["t2"]
        encoded_tree = request_result["encoding"]
        feedback = self.feedback_collector.collect_feedback([(t1, t2)])[0]
        self.provide_feedback(encoded_tree, feedback)

    def delayed_update(self, delay: float = 0.5):
        time.sleep(delay)
        self.immediate_update()
