from deeplearn.trainer.Trainer import Trainer


class StandardBatchTrainer(Trainer):
    def __init__(self, net, device, loss_fn, data, optimizer_name='adam',
                 is_classification_task=False, verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, batch_size=1, max_epochs=20):
        super(StandardBatchTrainer, self).__init__(net, device, data, optimizer_name, batch_size,
                                                   learning_rate, weight_decay, momentum, dampening,
                                                   None)
        self.is_classification_task = is_classification_task
        self.verbose = verbose
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs

    def train(self):
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
                outputs, _ = self.apply(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer_step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr
