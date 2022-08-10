from __future__ import annotations
import math
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Any


class NumericalData(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, scaler: Any = None):
        self.X = X  # numpy matrix of float values
        self.y = y  # numpy array
        self.scaler = scaler  # e.g., StandardScaler(), already fitted to the training data
        if not(scaler is None):
            self.X = scaler.transform(self.X)  # transform data
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()

    def __len__(self) -> int:
        # gets the number of rows in the dataset
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # gets a data point from the dataset as torch tensor array along with the label
        return self.X[idx], self.y[idx]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        dataset, labels = [], []
        for i in range(len(self)):
            curr_point, curr_label = self[i]
            dataset.append(curr_point.tolist())
            labels.append(curr_label.item())
        return np.array(dataset), np.array(labels)

    def remove_ground_truth_duplicates(self) -> NumericalData:
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
        return NumericalData(np.array(new_X, dtype=np.float32), np.array(new_y, dtype=np.float32), scaler=None)
