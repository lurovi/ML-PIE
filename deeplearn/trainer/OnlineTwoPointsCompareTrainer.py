from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from deeplearn.dataset.TreeData import TreeData
from deeplearn.dataset.TreeDataTwoPointsCompare import TreeDataTwoPointsCompare
from deeplearn.trainer.Trainer import Trainer
import torch.optim as optim
import torch
import numpy as np

from util.EvaluationMetrics import EvaluationMetrics

# TO DO
class OnlineTwoPointsCompareTrainer(Trainer):
    def __init__(self, net, device, dataloader,
                 verbose=False,
                 learning_rate=0.001, weight_decay=0.00001, momentum=0, dampening=0, max_epochs=20):
        super(OnlineTwoPointsCompareTrainer, self).__init__(net, device, dataloader)
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.max_epochs = max_epochs
        self.output_layer_size = net.number_of_output_neurons()
        self.input_layer_size = net.number_of_input_neurons()

    def train(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_epoch_arr = []
        one = torch.tensor(1, dtype=torch.float32)
        minus_one = torch.tensor(-1, dtype=torch.float32)
        for epoch in range(self.max_epochs):
            for batch in self.data:
                self.net.train()
                optimizer.zero_grad()
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
                single_point_dim = inputs.shape[1]//2
                inputs_1 = inputs[:, :single_point_dim]
                inputs_2 = inputs[:, single_point_dim:]
                outputs_1, _ = self.net(inputs_1)
                outputs_1 = outputs_1.flatten()
                loss_1 = one*torch.mean(labels*outputs_1)
                outputs_2, _ = self.net(inputs_2)
                outputs_2 = outputs_2.flatten()
                loss_2 = minus_one*torch.mean(labels*outputs_2)
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()
            loss_epoch_arr.append(loss.item())
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}. Loss: {loss.item()}.")
        return loss_epoch_arr

    def evaluate_classifier(self, dataloader):
        self.net.eval()
        y_true = []
        points = []
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float().reshape((labels.shape[0], 1))
            for i in range(len(inputs)):
                points.append(inputs[i].tolist())
                y_true.append(labels[i][0].item())
        y_true = np.array(y_true, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        ddd = TreeData(None, points, y_true, scaler=None)
        ddd = TreeDataTwoPointsCompare(ddd, 2000, binary_label=True)
        ddd = DataLoader(ddd, batch_size=1, shuffle=True)
        y_true = []
        y_pred = []
        self.net.eval()
        with torch.no_grad():
            for batch in ddd:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
                single_point_dim = inputs.shape[1] // 2
                inputs_1 = inputs[:, :single_point_dim]
                inputs_2 = inputs[:, single_point_dim:]
                outputs_1, _ = self.net(inputs_1)
                outputs_1 = outputs_1[0][0].item()
                outputs_2, _ = self.net(inputs_2)
                outputs_2 = outputs_2[0][0].item()
                if outputs_1 >= outputs_2:
                    pred = 1
                else:
                    pred = 0
                y_pred.append(pred)
                y_true.append(labels[0].item())
            cf_matrix = confusion_matrix(y_true, y_pred)
        self.net.train()
        return EvaluationMetrics.model_accuracy(cf_matrix)
