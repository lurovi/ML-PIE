import torch
import torch.nn as nn
import statistics
from typing import Any, List, Tuple


class MLPNet(nn.Module):
    def __init__(self, activation_func: Any, final_activation_func: Any, input_layer_size: int, output_layer_size: int, hidden_layer_sizes: List[int] = None, dropout_prob: float = 0.0):
        super(MLPNet, self).__init__()
        if hidden_layer_sizes is None:
            hidden_layer_sizes = []
        self.activation_func = activation_func  # e.g., nn.ReLU()
        self.final_activation_func = final_activation_func  # e.g., nn.Tanh()
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        fc_components = []
        layer_sizes = hidden_layer_sizes + [output_layer_size]
        curr_dim = input_layer_size
        for i in range(len(layer_sizes)):
            curr_layer = nn.Linear(curr_dim, layer_sizes[i])
            fc_components.append(curr_layer)
            if i != len(layer_sizes) - 1:
                fc_components.append(self.activation_func)
            else:
                fc_components.append(self.final_activation_func)
            if i == len(layer_sizes) - 3 or i == len(layer_sizes) - 5 or i == len(layer_sizes) - 7 or i == len(layer_sizes) - 9:
                fc_components.append(self.dropout)
            curr_dim = layer_sizes[i]

        self.fc_model = nn.Sequential(*fc_components[:-2])
        self.last_layer = nn.Sequential(fc_components[-2], fc_components[-1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        uncertainty = []
        x = self.fc_model(x)
        if self.training:
            x = self.last_layer(x)
        else:
            z = x.detach().clone()
            x = self.last_layer(x)
            uncert = [self.last_layer(nn.Dropout(self.dropout_prob)(z)) for _ in range(10)]
            for i in range(x.size(0)):
                curr_uncert = []
                for j in range(len(uncert)):
                    curr_uncert.append(uncert[j][i][0].item())
                uncertainty.append(statistics.pstdev(curr_uncert))
        return x, uncertainty

    def number_of_output_neurons(self) -> int:
        return self.output_layer_size

    def number_of_input_neurons(self) -> int:
        return self.input_layer_size
