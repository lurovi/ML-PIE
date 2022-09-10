import torch
import torch.nn as nn
import statistics
from typing import Any, List, Tuple


class MLPNet(nn.Module):
    def __init__(self, activation_func: Any,
                 final_activation_func: Any,
                 input_layer_size: int,
                 output_layer_size: int,
                 hidden_layer_sizes: List[int] = None,
                 dropout_prob: float = 0.0,
                 num_uncertainty_samples: int = 10):
        super().__init__()
        if hidden_layer_sizes is None:
            hidden_layer_sizes: List[int] = []
        self.__activation_func: Any = activation_func  # e.g., nn.ReLU()
        self.__final_activation_func: Any = final_activation_func  # e.g., nn.Tanh()
        self.__input_layer_size: int = input_layer_size
        self.__output_layer_size: int = output_layer_size
        self.__dropout_prob: float = dropout_prob
        self.__dropout: Any = nn.Dropout(self.__dropout_prob)
        self.__num_uncertainty_samples: int = num_uncertainty_samples

        fc_components: List = []
        layer_sizes: List = hidden_layer_sizes + [output_layer_size]
        curr_dim: int = input_layer_size
        for i in range(len(layer_sizes)):
            curr_layer = nn.Linear(curr_dim, layer_sizes[i])
            fc_components.append(curr_layer)
            if i != len(layer_sizes) - 1:
                fc_components.append(self.__activation_func)
            else:
                fc_components.append(self.__final_activation_func)
            if i == len(layer_sizes) - 3 or i == len(layer_sizes) - 5 or i == len(layer_sizes) - 7 or i == len(layer_sizes) - 9:
                fc_components.append(self.__dropout)
            curr_dim = layer_sizes[i]

        self.__fc_model: Any = nn.Sequential(*fc_components[:-2])
        self.__last_layer: Any = nn.Sequential(fc_components[-2], fc_components[-1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        uncertainty: List[float] = []
        x: torch.Tensor = self.__fc_model(x)
        if self.training:
            x = self.__last_layer(x)
        else:
            z: torch.Tensor = x.detach().clone()
            x = self.__last_layer(x)
            uncert: List = [self.__last_layer(nn.Dropout(self.__dropout_prob)(z)) for _ in range(self.__num_uncertainty_samples)]
            for i in range(x.size(0)):
                curr_uncert: List = []
                for j in range(len(uncert)):
                    curr_uncert.append(uncert[j][i][0].item())
                uncertainty.append(statistics.pstdev(curr_uncert))
        return x, uncertainty

    def number_of_output_neurons(self) -> int:
        return self.__output_layer_size

    def number_of_input_neurons(self) -> int:
        return self.__input_layer_size
