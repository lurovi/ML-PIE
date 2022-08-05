from deeplearn.trainer.Trainer import Trainer
import torch.optim as optim


class StandardBatchTrainer(Trainer):
    def __init__(self, net, device, loss_fn, data, optimizer_name='adam',
                 is_classification_task=False, verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, batch_size=1, max_epochs=20):
        super(StandardBatchTrainer, self).__init__(net, device, data, batch_size)
        self.is_classification_task = is_classification_task
        self.optimizer_name = optimizer_name
        self.verbose = verbose
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.max_epochs = max_epochs

    def train(self):
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(self.net_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(self.net_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                                  momentum=self.momentum, dampening=self.dampening)
        else:
            raise ValueError(f"{self.optimizer_name} is not a valid value for argument optimizer.")
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
                optimizer.zero_grad()
                outputs, _ = self.apply(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr
