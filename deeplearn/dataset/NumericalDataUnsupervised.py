from __future__ import annotations
import math
from collections import Counter

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Any, List


class NumericalDataUnsupervised(Dataset):
    def __init__(self, X: np.ndarray, scaler: Any = None):
        self.__X = X  # numpy matrix of float values
        self.__scaler = scaler  # e.g., StandardScaler(), already fitted to the training data
        if not(scaler is None):
            self.__X = scaler.transform(self.__X)  # transform data
        self.__X = torch.from_numpy(self.__X).float()

    def __len__(self) -> int:
        # gets the number of rows in the dataset
        return len(self.__X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # gets a data point from the dataset as torch tensor array along with the label
        return self.__X[idx]

    def all_indexes(self) -> List[int]:
        return list(range(len(self)))
