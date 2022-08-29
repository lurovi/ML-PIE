from deeplearn.trainer.TwoPointsCompareTrainer import TwoPointsCompareTrainer
from deeplearn.trainer.TrainerFactory import TrainerFactory

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from deeplearn.trainer.Trainer import Trainer


class TwoPointsCompareTrainerFactory(TrainerFactory):
    def __init__(self, verbose: bool = False, max_epochs: int = 20):
        super(TwoPointsCompareTrainerFactory, self).__init__()
        self.__verbose = verbose
        self.__max_epochs = max_epochs

    def create_trainer(self, net: nn.Module, device: torch.device, data: Dataset,
                       optimizer_name: str, batch_size: int = 1,
                       learning_rate: float = 0.001, weight_decay: float = 0.00001,
                       momentum: float = 0, dampening: float = 0,
                       custom_optimizer: torch.optim.Optimizer = None) -> Trainer:
        return TwoPointsCompareTrainer(net, device, data, self.__verbose,
                                       learning_rate, weight_decay, momentum, dampening,
                                       self.__max_epochs, batch_size)
