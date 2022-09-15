import threading

from torch.utils.data import Dataset

from deeplearn.trainer.Trainer import Trainer
from deeplearn.trainer.blockingtrainer.BlockingTrainer import BlockingTrainer


class BlockingOnlineTrainer(BlockingTrainer):
    def update(self, trainer: Trainer, data: Dataset, mutex: threading.Lock) -> None:
        with mutex:
            print("TRAIN")
            trainer.change_data(data)
            trainer.fit()
