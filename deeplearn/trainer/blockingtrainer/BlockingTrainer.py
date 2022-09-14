from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from deeplearn.trainer.Trainer import Trainer


class BlockingTrainer(ABC):

    @abstractmethod
    def update(self, trainer: Trainer, data: Dataset, mutex) -> None:
        pass
