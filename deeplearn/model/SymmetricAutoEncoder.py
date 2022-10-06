from functools import partial
import torch
import random
import numpy as np
from torch import nn
from typing import List, Any
from torch import optim

from deeplearn.dataset.NumericalDataUnsupervised import NumericalDataUnsupervised
from deeplearn.trainer.AutoEncoderBatchTrainer import AutoEncoderBatchTrainer
from deeplearn.trainer.StandardBatchTrainer import StandardBatchTrainer
from genepro import node_impl
from nsgp.encoder.CountsEncoder import CountsEncoder
from nsgp.encoder.LevelWiseCountsEncoder import LevelWiseCountsEncoder
from nsgp.encoder.OneHotEncoder import OneHotEncoder
from nsgp.structure.TreeStructure import TreeStructure


class SymmetricAutoEncoder(nn.Module):
    def __init__(self, activation_func: Any, final_activation_func: Any,
                 input_layer_size: int, bottleneck_size: int,
                 hidden_layer_sizes: List[int] = None, dropout_probability: float = 0.0):
        super().__init__()
        if hidden_layer_sizes is None:
            hidden_layer_sizes: List[int] = []
        self.__activation_func = activation_func  # e.g., nn.Tanh()
        self.__final_activation_func = final_activation_func  # e.g., nn.Sigmoid()
        self.__input_layer_size = input_layer_size
        self.__bottleneck_size = bottleneck_size
        self.__dropout_probability = dropout_probability
        self.__dropout = nn.Dropout(self.__dropout_probability)

        linear_layers = []
        layer_sizes = hidden_layer_sizes + [bottleneck_size]
        curr_dim = input_layer_size
        for i in range(len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                linear_layers.append(self.__dropout)
            linear_layers.append(nn.Linear(curr_dim, layer_sizes[i]))
            linear_layers.append(self.__activation_func)
            curr_dim = layer_sizes[i]
        self.__encoder = nn.Sequential(*linear_layers)

        linear_layers = []
        layer_sizes = [input_layer_size] + hidden_layer_sizes
        layer_sizes.reverse()
        curr_dim = bottleneck_size
        for i in range(len(layer_sizes)):
            linear_layers.append(nn.Linear(curr_dim, layer_sizes[i]))
            if i == 0:
                linear_layers.append(self.__activation_func)
                linear_layers.append(self.__dropout)
            elif i == len(layer_sizes) - 1:
                linear_layers.append(self.__final_activation_func)
            else:
                linear_layers.append(self.__activation_func)
            curr_dim = layer_sizes[i]
        self.__decoder = nn.Sequential(*linear_layers)

    def encode(self, x):
        return self.__encoder(x)

    def decode(self, x):
        return self.__decoder(x)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encode(x)
        x = self.decode(x)
        return x, [], None

    def number_of_output_neurons(self) -> int:
        return self.__input_layer_size

    def number_of_input_neurons(self) -> int:
        return self.__input_layer_size


if __name__ == "__main__":
    # Setting torch to use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    n_features = 7
    internal_nodes = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                      node_impl.UnaryMinus(), node_impl.Power(), node_impl.Square(), node_impl.Cube(),
                      node_impl.Sqrt(), node_impl.Exp(), node_impl.Log(), node_impl.Sin(), node_impl.Cos()]
    normal_distribution_parameters = [(0, 1), (0, 1), (0, 3), (0, 8), (0, 0.5), (0, 15), (0, 5), (0, 8), (0, 20),
                                      (0, 30), (0, 30), (0, 23), (0, 23)] + [(0, 0.8)] * n_features + [(0, 0.5)]
    structure = TreeStructure(internal_nodes, n_features, 5, ephemeral_func=partial(np.random.uniform, -5.0, 5.0),
                              normal_distribution_parameters=normal_distribution_parameters)
    encoder_onehot = OneHotEncoder(structure)
    bottleneck_size = int(structure.get_max_n_nodes()*2)
    encodings_0 = np.array([encoder_onehot.encode(structure.generate_tree()) for _ in range(10 ** 5)])
    encodings = NumericalDataUnsupervised(encodings_0)

    ae = SymmetricAutoEncoder(nn.Tanh(), nn.Identity(), encoder_onehot.size(), bottleneck_size, [1000, 600, 420, 200, 100], dropout_probability=0.10)
    trainer = AutoEncoderBatchTrainer(ae, device, encodings, verbose=True, batch_size=1000, max_epochs=30)
    trainer.fit()
    for tree_ind in [10, 40, 100, 50, 60]:
        print()
        example = torch.from_numpy(encodings_0[tree_ind]).float().reshape(1, -1)
        print(example[0].tolist())
        print(nn.Softmax(dim=1)(trainer.predict(example)[0][0].reshape(63, 21)).reshape(1, -1)[0].tolist())
        ae.eval()
        with torch.no_grad():
            print(ae.encode(example)[0].tolist())
        ae.train()
    print()
    print()
    print()
    encodings_0 = np.array([encoder_onehot.encode(structure.generate_tree()) for _ in range(10 ** 1)])
    for tree_ind in list(range(len(encodings_0))):
        print()
        example = torch.from_numpy(encodings_0[tree_ind]).float().reshape(1, -1)
        print(example[0].tolist())
        print(nn.Softmax(dim=1)(trainer.predict(example)[0][0].reshape(63, 21)).reshape(1, -1)[0].tolist())
        ae.eval()
        with torch.no_grad():
            print(ae.encode(example)[0].tolist())
        ae.train()
