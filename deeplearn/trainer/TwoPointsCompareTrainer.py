from typing import List

from deeplearn.trainer.Trainer import Trainer

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TwoPointsCompareTrainer(Trainer):
    def __init__(self, net: nn.Module,
                 device: torch.device,
                 data: Dataset,
                 verbose: bool = False,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.00001,
                 momentum: float = 0,
                 dampening: float = 0,
                 max_epochs: int = 20,
                 batch_size: int = 1):
        super().__init__(net=net, device=device, data=data, optimizer_name="adam", batch_size=batch_size,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         momentum=momentum, dampening=dampening, custom_optimizer=None)
        self.verbose: bool = verbose
        self.max_epochs: int = max_epochs

    def fit(self) -> List[float]:
        loss_epoch_arr = []
        loss = None
        one = torch.tensor(1, dtype=torch.float32)
        minus_one = torch.tensor(-1, dtype=torch.float32)
        self.set_train_mode()
        for epoch in range(self.max_epochs):
            for batch in self.all_batches():
                self.optimizer_zero_grad()
                inputs, labels = batch
                inputs, labels = self.to_device(inputs).float(), self.to_device(labels).float()
                single_point_dim = inputs.shape[1]//2
                inputs_1 = inputs[:, :single_point_dim]
                inputs_2 = inputs[:, single_point_dim:]
                outputs_1, _ = self.apply(inputs_1)
                outputs_1 = outputs_1.flatten()
                loss_1 = one*torch.sum(labels*outputs_1)
                outputs_2, _ = self.apply(inputs_2)
                outputs_2 = outputs_2.flatten()
                loss_2 = minus_one*torch.sum(labels*outputs_2)
                loss = loss_1 + loss_2
                loss.backward()
                self.optimizer_step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr
