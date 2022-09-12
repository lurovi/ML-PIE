from typing import List

from deeplearn.trainer.Trainer import Trainer
from deeplearn.trainer.TrainerFactory import TrainerFactory
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class OnlineTwoPointsCompareTrainer(Trainer):
    def __init__(self, net: nn.Module,
                 device: torch.device,
                 data: Dataset = None,
                 verbose: bool = False,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.00001,
                 momentum: float = 0,
                 dampening: float = 0,
                 warmup_trainer_factory: TrainerFactory = None,
                 warmup_dataset: Dataset = None):
        optimizer: torch.optim.Optimizer = None
        net_model: nn.Module = net
        if warmup_trainer_factory is not None and warmup_dataset is not None:
            pre_trainer: Trainer = warmup_trainer_factory.create_trainer(net=net_model, device=device,
                                                                         data=warmup_dataset, optimizer_name="adam",
                                                                         batch_size=1, learning_rate=learning_rate,
                                                                         weight_decay=weight_decay,
                                                                         momentum=momentum, dampening=dampening,
                                                                         custom_optimizer=None)
            pre_trainer.fit()
            net_model = pre_trainer.get_net()
            optimizer = pre_trainer.get_optimizer()
        super().__init__(net=net_model, device=device, data=data, optimizer_name="adam", batch_size=1,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         momentum=momentum, dampening=dampening, custom_optimizer=optimizer)
        if data is not None and len(data) != 1:
            raise AttributeError("Online training requires a training set with exactly one record at time.")
        self.verbose: bool = verbose

    def change_data(self, data: Dataset) -> None:
        if len(data) != 1:
            raise AttributeError("Online training requires a training set with exactly one record at time.")
        super().change_data(data)

    def fit(self) -> List[float]:
        loss_epoch_arr = []
        one = torch.tensor(1, dtype=torch.float32)
        minus_one = torch.tensor(-1, dtype=torch.float32)
        self.set_train_mode()
        self.optimizer_zero_grad()
        inputs, labels = self.all_batches()[0]
        inputs, labels = self.to_device(inputs).float(), self.to_device(labels).float()
        single_point_dim = inputs.shape[1] // 2
        inputs_1 = inputs[:, :single_point_dim]
        inputs_2 = inputs[:, single_point_dim:]
        outputs_1, _, _ = self.apply(inputs_1)
        outputs_1 = outputs_1.flatten()
        loss_1 = one * torch.sum(labels * outputs_1)
        outputs_2, _, _ = self.apply(inputs_2)
        outputs_2 = outputs_2.flatten()
        loss_2 = minus_one * torch.sum(labels * outputs_2)
        loss = loss_1 + loss_2
        loss.backward()
        self.optimizer_step()
        loss_epoch_arr.append(loss.item())
        if self.verbose:
            print(f"Loss: {loss.item()}.")
        return loss_epoch_arr
