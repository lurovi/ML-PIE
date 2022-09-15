import threading

from torch.utils.data import Dataset

from deeplearn.trainer.Trainer import Trainer
from deeplearn.trainer.blockingtrainer.BlockingTrainer import BlockingTrainer


class BlockingBatchTrainer(BlockingTrainer):

    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.dataset = []

    def update(self, trainer: Trainer, data: Dataset, mutex: threading.Lock) -> None:
        self.dataset.append(data)
        if len(self.dataset) >= self.batch_size:
            with mutex:
                for d in self.dataset:
                    trainer.change_data(d)
                    trainer.fit()
            self.dataset.clear()
