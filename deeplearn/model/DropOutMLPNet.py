import torch
import torch.nn as nn
import statistics
from typing import Any, List, Tuple


class DropOutMLPNet(nn.Module):
    def __init__(self, activation_func: Any,
                 final_activation_func: Any,
                 input_layer_size: int,
                 dropout_prob_uncertainty: float = 0.25,
                 num_uncertainty_samples: int = 10):
        super().__init__()
        self.__activation_func: Any = activation_func  # e.g., nn.ReLU()
        self.__final_activation_func: Any = final_activation_func  # e.g., nn.Tanh()
        self.__input_layer_size: int = input_layer_size
        self.__output_layer_size: int = 1
        self.__dropout_prob_uncertainty: float = dropout_prob_uncertainty
        self.__num_uncertainty_samples: int = num_uncertainty_samples

        self.__first_layer = nn.Linear(self.__input_layer_size, 100)
        self.__second_layer = nn.Linear(100, 100)
        self.__third_layer = nn.Linear(100, 100)
        self.__output_layer = nn.Linear(100, 1)

    def __apply_layers(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__first_layer(x)
        x = nn.Dropout(self.__dropout_prob_uncertainty)(x)
        x = self.__activation_func(x)

        x = self.__second_layer(x)
        x = nn.Dropout(self.__dropout_prob_uncertainty)(x)
        x = self.__activation_func(x)

        x = self.__third_layer(x)
        x = nn.Dropout(self.__dropout_prob_uncertainty)(x)
        x = self.__activation_func(x)

        x = self.__output_layer(x)
        x = self.__final_activation_func(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        uncertainty: List[float] = []
        predictions: List[float] = []
        if self.training:
            out: torch.Tensor = self.__apply_layers(x)
        else:
            z: torch.Tensor = x.detach().clone()
            uncert: List = [self.__apply_layers(z) for _ in range(self.__num_uncertainty_samples)]
            for i in range(x.size(0)):
                curr_uncert: List = []
                for j in range(len(uncert)):
                    curr_uncert.append(uncert[j][i][0].item())
                uncertainty.append(statistics.pstdev(curr_uncert))
                predictions.append(statistics.mean(curr_uncert))
            out: torch.Tensor = torch.tensor(predictions, dtype=torch.float32).reshape(-1, 1)
        return out, uncertainty, torch.tensor(1, dtype=torch.float32)

    def number_of_output_neurons(self) -> int:
        return self.__output_layer_size

    def number_of_input_neurons(self) -> int:
        return self.__input_layer_size
