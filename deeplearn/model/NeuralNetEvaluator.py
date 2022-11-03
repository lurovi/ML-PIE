import torch
import torch.nn as nn
from sklearn.metrics import r2_score, confusion_matrix
from torch.utils.data import DataLoader

from util.EvaluationMetrics import EvaluationMetrics
from util.Sort import Sort

from typing import Tuple, Dict


class NeuralNetEvaluator:

    @staticmethod
    def evaluate_softmax_classification_accuracy_f1(net: nn.Module,
                                                    dataloader: DataLoader,
                                                    device: torch.device) -> Tuple[float, Dict[int, Dict[str, float]]]:
        y_true = []
        y_pred = []
        net.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                outputs, _, _ = net(inputs)
                pred = outputs.data.argmax(dim=1).float()
                y_pred.extend(pred.tolist())
                y_true.extend(labels.tolist())
            cf_matrix = confusion_matrix(y_true, y_pred)
        net.train()
        return EvaluationMetrics.model_accuracy(cf_matrix)

    @staticmethod
    def evaluate_pairs_classification_accuracy_with_siso_net(net: nn.Module,
                                                             dataloader: DataLoader,
                                                             device: torch.device) -> float:
        y_true = []
        y_pred = []
        net.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                single_point_dim = inputs.shape[1] // 2
                inputs_1 = inputs[:, :single_point_dim]
                inputs_2 = inputs[:, single_point_dim:]
                outputs_1, _, _ = net(inputs_1)
                outputs_1 = outputs_1.flatten()
                outputs_2, _, _ = net(inputs_2)
                outputs_2 = outputs_2.flatten()
                pred = []
                for i in range(len(outputs_1)):
                    if outputs_1[i] >= outputs_2[i]:
                        pred.append(-1.0)
                    else:
                        pred.append(1.0)
                y_pred.extend(pred)
                y_true.extend(labels.tolist())
            cf_matrix = confusion_matrix(y_true, y_pred)
        net.train()
        return EvaluationMetrics.model_accuracy(cf_matrix)[0]

    @staticmethod
    def evaluate_regression(net: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
        y_true = []
        y_pred = []
        net.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                outputs, _, _ = net(inputs)
                y_pred.extend(outputs.flatten().tolist())
                y_true.extend(labels.tolist())
        net.train()
        return r2_score(y_true, y_pred)

    @staticmethod
    def evaluate_ranking(net: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
        y_true = []
        y_pred = []
        net.eval()
        idx = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                outputs, _, _ = net(inputs)
                for i in range(len(labels)):
                    curr_input, curr_label, curr_output = inputs[i], labels[i].item(), outputs[i][0].item()
                    y_true.append((curr_input, curr_label, idx))
                    y_pred.append((curr_input, curr_output, idx))
                    idx += 1
            y_true, _ = Sort.heapsort(y_true, lambda a, b: a[1] < b[1], inplace=False, reverse=False)
            y_pred, _ = Sort.heapsort(y_pred, lambda a, b: a[1] < b[1], inplace=False, reverse=False)
        net.train()
        return EvaluationMetrics.spearman_footrule(y_true, y_pred, lambda a, b: a[2] == b[2])

    @staticmethod
    def evaluate_average_uncertainty(net: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
        y_pred = []
        net.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                _, uncertainties, _ = net(inputs)
                for i in range(len(labels)):
                    curr_input, curr_label, curr_uncert = inputs[i], labels[i].item(), uncertainties[i]
                    y_pred.append(curr_uncert)
            aaa = sum(y_pred) / len(y_pred)
        net.train()
        return aaa
