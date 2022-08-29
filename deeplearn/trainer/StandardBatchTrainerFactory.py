from typing import Any

from deeplearn.trainer.StandardBatchTrainer import StandardBatchTrainer
from deeplearn.trainer.TrainerFactory import TrainerFactory

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from deeplearn.trainer.Trainer import Trainer


class StandardBatchTrainerFactory(TrainerFactory):
    def __init__(self, loss_fn: Any, is_classification_task: bool = False, verbose: bool = False, max_epochs: int = 20):
        super(StandardBatchTrainerFactory, self).__init__()
        self.__loss_fn = loss_fn
        self.__is_classification_task = is_classification_task
        self.__verbose = verbose
        self.__max_epochs = max_epochs

    def create_trainer(self, net: nn.Module, device: torch.device, data: Dataset,
                       optimizer_name: str, batch_size: int = 1,
                       learning_rate: float = 0.001, weight_decay: float = 0.00001,
                       momentum: float = 0, dampening: float = 0,
                       custom_optimizer: torch.optim.Optimizer = None) -> Trainer:
        return StandardBatchTrainer(net, device, self.__loss_fn, data, optimizer_name,
                                    self.__is_classification_task, self.__verbose,
                                    learning_rate, weight_decay, momentum, dampening,
                                    batch_size, self.__max_epochs)
