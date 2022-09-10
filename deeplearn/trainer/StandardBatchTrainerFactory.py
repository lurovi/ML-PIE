from typing import Callable

from deeplearn.trainer.StandardBatchTrainer import StandardBatchTrainer
from deeplearn.trainer.TrainerFactory import TrainerFactory

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from deeplearn.trainer.Trainer import Trainer


class StandardBatchTrainerFactory(TrainerFactory):
    def __init__(self, loss_fn: Callable,
                 is_classification_task: bool = False,
                 verbose: bool = False,
                 max_epochs: int = 20):
        super().__init__()
        self.__loss_fn: Callable = loss_fn
        self.__is_classification_task: bool = is_classification_task
        self.__verbose: bool = verbose
        self.__max_epochs: int = max_epochs

    def create_trainer(self, net: nn.Module, device: torch.device, data: Dataset,
                       optimizer_name: str, batch_size: int = 1,
                       learning_rate: float = 0.001, weight_decay: float = 0.00001,
                       momentum: float = 0, dampening: float = 0,
                       custom_optimizer: torch.optim.Optimizer = None) -> Trainer:
        return StandardBatchTrainer(net=net, device=device, loss_fn=self.__loss_fn, data=data,
                                    optimizer_name=optimizer_name,
                                    is_classification_task=self.__is_classification_task,
                                    verbose=self.__verbose,
                                    learning_rate=learning_rate, weight_decay=weight_decay,
                                    momentum=momentum, dampening=dampening,
                                    batch_size=batch_size, max_epochs=self.__max_epochs)
