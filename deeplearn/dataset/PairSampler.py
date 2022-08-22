import random
import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np

from deeplearn.trainer.Trainer import Trainer
from util.Sort import Sort


class PairSampler:

    @staticmethod
    def random_sampler_with_replacement(X: torch.Tensor, y: torch.Tensor, n_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
        train_indexes = list(range(len(y)))
        X_pairs, y_pairs = [], []
        for _ in range(n_pairs):
            idx_1 = random.choice(train_indexes)
            first_point, first_label = X[idx_1], y[idx_1].item()
            exit_loop = False
            while not (exit_loop):
                idx_2 = random.choice(train_indexes)
                if idx_2 != idx_1:
                    exit_loop = True
                    second_point, second_label = X[idx_2], y[idx_2].item()
            if first_label >= second_label:
                curr_feedback = -1
            else:
                curr_feedback = 1
            curr_point = first_point.tolist() + second_point.tolist()
            y_pairs.append(curr_feedback)
            X_pairs.append(curr_point)
        return np.array(X_pairs, dtype=np.float32), np.array(y_pairs, dtype=np.float32)

    @staticmethod
    def random_sampler(X: torch.Tensor, y: torch.Tensor, already_seen: List[int], n_pairs: int) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        train_indexes = list(range(len(y)))
        X_pairs, y_pairs = [], []
        for _ in range(n_pairs):
            exit_loop = False
            while not (exit_loop):
                idx_1 = random.choice(train_indexes)
                if idx_1 not in already_seen:
                    exit_loop = True
                    already_seen.append(idx_1)
                    first_point, first_label = X[idx_1], y[idx_1].item()
            exit_loop = False
            while not (exit_loop):
                idx_2 = random.choice(train_indexes)
                if idx_2 != idx_1 and idx_2 not in already_seen:
                    exit_loop = True
                    already_seen.append(idx_2)
                    second_point, second_label = X[idx_2], y[idx_2].item()
            if first_label >= second_label:
                curr_feedback = -1
            else:
                curr_feedback = 1
            curr_point = first_point.tolist() + second_point.tolist()
            y_pairs.append(curr_feedback)
            X_pairs.append(curr_point)
        return np.array(X_pairs, dtype=np.float32), np.array(y_pairs, dtype=np.float32), already_seen

    @staticmethod
    def random_sampler_online(X: torch.Tensor, y: torch.Tensor, already_seen: List[int]) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        train_indexes = list(range(len(y)))
        exit_loop = False
        while not (exit_loop):
            idx_1 = random.choice(train_indexes)
            if idx_1 not in already_seen:
                exit_loop = True
                already_seen.append(idx_1)
                first_point, first_label = X[idx_1], y[idx_1].item()
        exit_loop = False
        while not (exit_loop):
            idx_2 = random.choice(train_indexes)
            if idx_2 != idx_1 and idx_2 not in already_seen:
                exit_loop = True
                already_seen.append(idx_2)
                second_point, second_label = X[idx_2], y[idx_2].item()
        if first_label >= second_label:
            curr_feedback = np.array([-1], dtype=np.float32)
        else:
            curr_feedback = np.array([1], dtype=np.float32)
        curr_point = np.array([first_point.tolist() + second_point.tolist()], dtype=np.float32)
        return curr_point, curr_feedback, already_seen

    @staticmethod
    def uncertainty_sampler_online(X: torch.Tensor, y: torch.Tensor, trainer: Trainer, already_seen: List[int]) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        _, uncertainty = trainer.predict(X)
        _, ind_points = Sort.heapsort(uncertainty, lambda a, b: a < b, inplace=False, reverse=True)
        count = 0
        i = 0
        points = []
        while count < 2 and i < len(ind_points):
            if ind_points[i] not in already_seen:
                already_seen.append(ind_points[i])
                count += 1
                points.append((X[ind_points[i]], y[ind_points[i]].item()))
            i += 1
        first_point, first_label = points[0]
        second_point, second_label = points[1]
        if first_label >= second_label:
            curr_feedback = np.array([-1], dtype=np.float32)
        else:
            curr_feedback = np.array([1], dtype=np.float32)
        curr_point = np.array([first_point.tolist() + second_point.tolist()], dtype=np.float32)
        return curr_point, curr_feedback, already_seen
