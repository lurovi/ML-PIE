from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer
from deeplearn.trainer.TrainerFactory import TrainerFactory

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from deeplearn.trainer.Trainer import Trainer


class OnlineTwoPointsCompareTrainerFactory(TrainerFactory):
    def __init__(self, verbose: bool = False,
                 warmup_trainer_factory: TrainerFactory = None,
                 warmup_dataset: Dataset = None):
        super().__init__()
        self.__verbose: bool = verbose
        self.__warmup_trainer_factory: TrainerFactory = warmup_trainer_factory
        self.__warmup_dataset: Dataset = warmup_dataset

    def create_trainer(self, net: nn.Module, device: torch.device, data: Dataset,
                       optimizer_name: str, batch_size: int = 1,
                       learning_rate: float = 0.001, weight_decay: float = 0.00001,
                       momentum: float = 0, dampening: float = 0,
                       custom_optimizer: torch.optim.Optimizer = None) -> Trainer:
        return OnlineTwoPointsCompareTrainer(net=net, device=device, data=data, verbose=self.__verbose,
                                             learning_rate=learning_rate, weight_decay=weight_decay,
                                             momentum=momentum, dampening=dampening,
                                             warmup_trainer_factory=self.__warmup_trainer_factory,
                                             warmup_dataset=self.__warmup_dataset)
