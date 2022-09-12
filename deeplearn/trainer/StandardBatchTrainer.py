from typing import Callable, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from deeplearn.trainer.Trainer import Trainer


class StandardBatchTrainer(Trainer):
    def __init__(self, net: nn.Module,
                 device: torch.device,
                 loss_fn: Callable,
                 data: Dataset,
                 optimizer_name: str = 'adam',
                 is_classification_task: bool = False,
                 verbose: bool = False,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.00001,
                 momentum: float = 0,
                 dampening: float = 0,
                 batch_size: int = 1,
                 max_epochs: int = 20):
        super().__init__(net=net, device=device, data=data, optimizer_name=optimizer_name, batch_size=batch_size,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         momentum=momentum, dampening=dampening, custom_optimizer=None)
        self.is_classification_task: bool = is_classification_task
        self.verbose: bool = verbose
        self.loss_fn: Callable = loss_fn
        self.max_epochs: int = max_epochs

    def fit(self) -> List[float]:
        loss_epoch_arr = []
        loss = None
        self.set_train_mode()
        for epoch in range(self.max_epochs):
            for batch in self.all_batches():
                inputs, labels = batch
                inputs = self.to_device(inputs).float()
                if self.is_classification_task:
                    labels = self.to_device(labels).long()
                else:
                    labels = self.to_device(labels).float().reshape((labels.shape[0], 1))
                self.optimizer_zero_grad()
                outputs, _, _ = self.apply(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer_step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr
