from __future__ import annotations
import math
from collections import Counter

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Any, List


class NumericalDataFromTorchTensor(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, scaler: Any = None):
        self.__X = X  # matrix of float values
        self.__y = y  # tensor
        self.__scaler = scaler  # e.g., StandardScaler(), already fitted to the training data
        if not(scaler is None):
            self.__X = scaler.transform(self.__X.numpy())  # transform data
            self.__X = torch.from_numpy(self.__X)
        self.__X = self.__X.float()
        self.__y = self.__y.float()

    def __len__(self) -> int:
        # gets the number of rows in the dataset
        return len(self.__y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # gets a data point from the dataset as torch tensor array along with the label
        return self.__X[idx], self.__y[idx]

    def subset(self, indexes: List[int]) -> NumericalDataFromTorchTensor:
        X, y = [], []
        for i in indexes:
            if not (0 <= i < len(self)):
                raise IndexError(f"{i} is out of range as index for this dataset.")
            point, label = self[i]
            X.append(point.tolist())
            y.append(label.item())
        return NumericalDataFromTorchTensor(torch.tensor(X), torch.tensor(y))

    def get_points_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = [], []
        for i in self.all_indexes():
            point, label = self[i]
            X.append(point.tolist())
            y.append(label.item())
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def all_indexes(self) -> List[int]:
        return list(range(len(self)))

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        dataset, labels = [], []
        for i in range(len(self)):
            curr_point, curr_label = self[i]
            dataset.append(curr_point.tolist())
            labels.append(curr_label.item())
        return np.array(dataset), np.array(labels)

    def remove_ground_truth_duplicates(self) -> NumericalDataFromTorchTensor:
        new_X, new_y = [], []
        for i in range(len(self)):
            is_dupl = False
            j = i+1
            exit_loop = False
            while not(exit_loop) and j < len(self):
                if math.isclose(self[i][1].item(), self[j][1].item(), rel_tol=1e-5):
                    is_dupl = True
                    exit_loop = True
                j += 1
            if not(is_dupl):
                new_X.append(self[i][0].tolist())
                new_y.append(self[i][1].item())
        return NumericalDataFromTorchTensor(torch.tensor(new_X), torch.tensor(new_y), scaler=None)

    def count_labels(self) -> Counter:
        return Counter(self.__y.tolist())
