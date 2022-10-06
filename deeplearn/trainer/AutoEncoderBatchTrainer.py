from typing import Callable, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from deeplearn.trainer.Trainer import Trainer


class AutoEncoderBatchTrainer(Trainer):
    def __init__(self, net: nn.Module,
                 device: torch.device,
                 data: Dataset,
                 optimizer_name: str = 'adam',
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
        self.verbose: bool = verbose
        self.loss_fn: Callable = AutoEncoderBatchTrainer.one_hot_multiple_cross_entropy_loss  # nn.MSELoss(reduction="mean")
        self.max_epochs: int = max_epochs

    def fit(self) -> List[float]:
        loss_epoch_arr = []
        loss = None
        self.set_train_mode()
        for epoch in range(self.max_epochs):
            for batch in self.all_batches():
                inputs = batch
                inputs = self.to_device(inputs).float()
                self.optimizer_zero_grad()
                outputs, _, _ = self.apply(inputs)
                loss = self.loss_fn(outputs, inputs)
                loss.backward()
                self.optimizer_step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr

    @staticmethod
    def one_hot_multiple_cross_entropy_loss(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        k = 21
        n = 63
        d = n * k
        m = inputs.shape[0]
        molt = 1.0 / float(m)
        inputs = inputs.reshape((m, n, k))
        outputs = outputs.reshape((m, n, k))
        outputs = nn.Softmax(dim=1)(outputs)
        outputs = torch.log(outputs + 1e-9)
        res = -1.0 * torch.sum(torch.mul(inputs, outputs), dim=1)
        res = res.reshape((m, -1))
        return molt * torch.sum(res, dim=1).sum()

    @staticmethod
    def one_hot_log_approximation_loss(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        molt = 1.0/float(inputs.shape[0])
        res = torch.add(torch.mul(inputs*21, torch.log(1.0 - outputs + 1e-9)), torch.mul(1.0 - inputs, torch.log(outputs + 1e-9)))
        return molt * torch.sum(res, dim=1).sum()
