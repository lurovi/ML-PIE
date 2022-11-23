import torch
import torch.nn as nn
import statistics
from typing import Any, List, Tuple

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.trainer.StandardBatchTrainer import StandardBatchTrainer


class MLPNet(nn.Module):
    def __init__(self,
                 activation_func: Any,
                 final_activation_func: Any,
                 input_layer_size: int,
                 output_layer_size: int,
                 hidden_layer_sizes: List[int] = None,
                 dropout_prob: float = 0.0,
                 dropout_in_train: bool = False,
                 dropout_in_eval: bool = False,
                 num_uncertainty_samples: int = 10):
        super().__init__()
        if hidden_layer_sizes is None:
            hidden_layer_sizes: List[int] = []
        self.__activation_func: Any = activation_func  # e.g., nn.ReLU()
        self.__final_activation_func: Any = final_activation_func  # e.g., nn.Tanh()
        self.__input_layer_size: int = input_layer_size
        self.__output_layer_size: int = output_layer_size
        self.__dropout_prob: float = dropout_prob
        self.__num_uncertainty_samples: int = num_uncertainty_samples

        self.__fc_components: List[nn.Linear] = []
        layer_sizes: List[int] = hidden_layer_sizes + [output_layer_size]
        curr_dim: int = input_layer_size
        for i in range(len(layer_sizes)):
            curr_layer: nn.Linear = nn.Linear(curr_dim, layer_sizes[i])
            self.__fc_components.append(curr_layer)
            curr_dim = layer_sizes[i]

        self.__fc_model: nn.Sequential = nn.Sequential(*self.__fc_components)
        self.__is_single_output: bool = True if self.__output_layer_size == 1 else False
        self.__dropout_in_train: bool = dropout_in_train
        self.__dropout_in_eval: bool = dropout_in_eval

    def __apply_layers(self, x: torch.Tensor, is_in_train_mode: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        z: torch.Tensor = None

        for i in range(len(self.__fc_components)):
            x = self.__fc_model[i](x)
            if i != len(self.__fc_components) - 1:
                if i == len(self.__fc_components) - 2:
                    z = x.detach().clone()
                    z = self.__activation_func(z)
                if is_in_train_mode:
                    if self.__dropout_in_train:
                        x = nn.Dropout(self.__dropout_prob)(x)
                else:
                    if self.__dropout_in_eval:
                        x = nn.Dropout(self.__dropout_prob)(x)
                x = self.__activation_func(x)
            else:
                x = self.__final_activation_func(x)

        return x, z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        uncertainties: List[float] = []
        predictions: List[float] = []
        is_in_train_mode: bool = self.training
        if is_in_train_mode:
            out_t: Tuple[torch.Tensor, torch.Tensor] = self.__apply_layers(x, is_in_train_mode=is_in_train_mode)
            out: torch.Tensor = out_t[0]
            embedding: torch.Tensor = out_t[1]
        else:
            z: torch.Tensor = x.detach().clone()
            out_t: Tuple[torch.Tensor, torch.Tensor] = self.__apply_layers(z, is_in_train_mode=is_in_train_mode)
            out: torch.Tensor = out_t[0]
            embedding: torch.Tensor = out_t[1]
            if self.__is_single_output:
                multiple_predictions: List[torch.Tensor] = [self.__apply_layers(z, is_in_train_mode=is_in_train_mode)[0]
                                                            for _ in range(self.__num_uncertainty_samples)]
                for input_index in range(x.size(0)):
                    multiple_predictions_for_current_input: List[float] = []
                    for prediction_index in range(len(multiple_predictions)):
                        multiple_predictions_for_current_input.append(
                            multiple_predictions[prediction_index][input_index][0].item()
                        )
                    mu: float = statistics.mean(multiple_predictions_for_current_input)
                    uncertainty: float = statistics.pstdev(multiple_predictions_for_current_input, mu)
                    uncertainties.append(uncertainty)
                    predictions.append(mu)
                out: torch.Tensor = torch.tensor(predictions, dtype=torch.float32).reshape(-1, 1)
            else:
                out_softmax: torch.Tensor = out.detach().clone()
                softmax_func: nn.Softmax = nn.Softmax(dim=1)
                confidences: List[float] = softmax_func(out_softmax).std(dim=1, unbiased=False).flatten().tolist()
                uncertainties = [1.0 - c for c in confidences]
        return out, uncertainties, embedding

    def number_of_output_neurons(self) -> int:
        return self.__output_layer_size

    def number_of_input_neurons(self) -> int:
        return self.__input_layer_size


if __name__ == "__main__":
    net: nn.Module = MLPNet(nn.ReLU(), nn.Identity(), 50, 1, [], 0.25, True, False, 10)
    X: torch.Tensor = torch.randint(low=-5, high=5, size=(50000, 50))
    y: torch.Tensor = X.sum(dim=1)
    print(net)
    dataset = NumericalData(X.numpy(), y.numpy())
    trainer = StandardBatchTrainer(net, torch.device("cpu"), nn.MSELoss(reduction="mean"),
                                   dataset, verbose=True, batch_size=100)
    trainer.fit()
