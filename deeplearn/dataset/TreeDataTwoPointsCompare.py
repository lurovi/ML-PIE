from __future__ import annotations
import math
import torch
import random
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

from deeplearn.dataset.TreeData import TreeData


class TreeDataTwoPointsCompare(Dataset):
    def __init__(self, dataset: TreeData, number_of_points: int, binary_label: bool = False, criterion: str = "target"):
        self.dataset = dataset  # this is an instance of a class that inherits torch.utils.data.Dataset class
        self.number_of_points = number_of_points  # how many points we want to generate
        self.original_number_of_points = len(dataset)  # the number of points in dataset instance
        new_data = []
        new_labels = []
        original_data = []
        original_labels = []
        original_trees = []
        indexes = list(range(self.original_number_of_points))  # [0, 1, 2, ..., self.original_number_of_points-2, self.original_number_of_points-1]
        for _ in range(self.number_of_points):
            exit_loop = False
            while not(exit_loop):
                idx = random.choices(indexes, k=2)  # extract two points at random with replacement (for computational efficiency reasons)
                first_point, first_label = self.dataset[idx[0]]  # first point extracted
                second_point, second_label = self.dataset[idx[1]]  # second point extracted
                if criterion == "target":
                    if not(math.isclose(first_label.item(), second_label.item(), rel_tol=1e-5)):  # if ground truths are equals, then sample again the two points
                        exit_loop = True
                elif criterion == "nodes":
                    if not(math.isclose(first_label.item(), second_label.item(), rel_tol=1e-5))\
                            and \
                            abs(self.dataset.get_tree(idx[0]).number_of_nodes() - self.dataset.get_tree(idx[1]).number_of_nodes() ) <= 4:
                        exit_loop = True
                else:
                    raise AttributeError(f"{criterion} is not a valid criterion for TreeDataTwoPointsCompare constructor.")
            original_data.append(first_point.tolist())
            original_data.append(second_point.tolist())
            if self.dataset.trees is not None:
                original_trees.append(self.dataset.get_tree(idx[0]))
                original_trees.append(self.dataset.get_tree(idx[1]))
            original_labels.append(first_label.item())
            original_labels.append(second_label.item())
            if first_label.item() >= second_label.item():  # first point has a higher score than the second one
                if binary_label:
                    new_labels.append(1)  # close to one when the first point is higher: sigmoid(z_final) >= 0.5
                else:
                    new_labels.append(-1.0)  # if the first point is higher, then the loss decreases: -1*(p1-p2)
            else:  # first point has a lower score than the second one
                if binary_label:
                    new_labels.append(0)  # close to zero when the first point is lower: sigmoid(z_final) < 0.5
                else:
                    new_labels.append(1.0)  # if the second point is higher, then the loss decreases: 1*(p1-p2)
            new_data.append(first_point.tolist() + second_point.tolist())
        self.X = torch.tensor(new_data).float()
        self.y = torch.tensor(new_labels).float()
        self.original_X = np.array(original_data, dtype=np.float32)
        self.original_y = np.array(original_labels, dtype=np.float32)
        self.original_trees = original_trees

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

    def to_simple_torch_dataset(self) -> TreeData:
        return TreeData(self.original_trees, self.original_X, self.original_y, scaler=None)
