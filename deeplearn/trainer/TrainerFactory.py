from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from deeplearn.trainer.Trainer import Trainer


class TrainerFactory(ABC):

    @abstractmethod
    def create_trainer(self, net: nn.Module, device: torch.device, data: Dataset,
                       optimizer_name: str, batch_size: int = 1,
                       learning_rate: float = 0.001, weight_decay: float = 0.00001,
                       momentum: float = 0, dampening: float = 0,
                       custom_optimizer: torch.optim.Optimizer = None) -> Trainer:
        pass
