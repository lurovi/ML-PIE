import random
from typing import List, Dict, Tuple

import pandas as pd

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer

from copy import deepcopy

from deeplearn.dataset.NumericalData import NumericalData
from deeplearn.dataset.PairSampler import PairSampler
from genepro.node import Node

from genepro.util import tree_from_prefix_repr

import numpy as np

from nsgp.structure.TreeStructure import TreeStructure
from util.PicklePersist import PicklePersist


class DatasetGenerator:
    def __init__(self, folder_path: str,
                 structure: TreeStructure,
                 training_size: int,
                 validation_size: int,
                 test_size: int,
                 seed: int = None):
        self.__seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.__structure = structure
        self.__folder_path = folder_path
        self.__training_size = training_size
        self.__validation_size = validation_size
        self.__test_size = test_size
        self.__train = [self.__structure.generate_tree() for _ in range(training_size)]
        self.__val = [self.__structure.generate_tree() for _ in range(validation_size)]
        self.__test = [self.__structure.generate_tree() for _ in range(test_size)]
        '''
        {
        "encoding_type_1": {
                                "training": numpy matrix,
                                "validation": numpy matrix,
                                "test": numpy matrix,
                           },
        "encoding_type_2": {
                                "training": numpy matrix,
                                "validation": numpy matrix,
                                "test": numpy matrix,
                           },    
        ...
        }
        '''
        self.__data_encoded: Dict[str, Dict[str, np.ndarray]] = {}
        '''
        {
        "ground_truth_1": {
                                "training": numpy array,
                                "validation": numpy array,
                                "test": numpy array,
                           },
        "ground_truth_2": {
                                "training": numpy array,
                                "validation": numpy array,
                                "test": numpy array,
                           },    
        ...
        }
        '''
        self.__ground_truths: Dict[str, Dict[str, np.ndarray]] = {}
        '''
        {
        "encoding_type_1": {
                                "ground_truth_1": NumericalData,
                                "ground_truth_2": NumericalData,
                                ... 
                           },
        ...
        }
        '''
        self.__warm_up_data: Dict[str, Dict[str, NumericalData]] = {}

    def shape(self) -> Tuple[int, int, int]:
        return self.__training_size, self.__validation_size, self.__test_size

    def original_trees(self) -> Tuple[List[Node], List[Node], List[Node]]:
        return deepcopy(self.__train), deepcopy(self.__val), deepcopy(self.__test)

    def get_X_y(self, encoding_type: str, ground_truth_type: str) -> Dict[str, NumericalData]:
        if encoding_type not in self.__data_encoded.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        if ground_truth_type not in self.__ground_truths.keys():
            raise AttributeError(f"{ground_truth_type} is not a valid ground truth type.")
        X = self.__data_encoded[encoding_type]
        y = self.__ground_truths[ground_truth_type]
        df = {}
        for s in ["training", "validation", "test"]:
            df[s] = NumericalData(X[s], y[s])
        return df

    def get_training_pairs(self, encoding_type: str, ground_truth_type: str, n_pairs: int, replacement: bool = False) -> NumericalData:
        X, y = self.get_X_y(encoding_type, ground_truth_type)["training"].get_points_and_labels()
        if replacement:
            X, y = PairSampler.random_sampler_with_replacement(X, y, n_pairs)
        else:
            X, y, _ = PairSampler.random_sampler(X, y, [], n_pairs)
        return NumericalData(X, y)

    def get_seed(self) -> int:
        return self.__seed

    def set_seed(self, seed) -> None:
        self.__seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def persist(self, file_name: str) -> None:
        PicklePersist.compress_pickle(self.__folder_path + "/" + file_name, self)

    def get_folder_path(self) -> str:
        return self.__folder_path

    def get_structure(self) -> TreeStructure:
        return self.__structure

    def generate_tree_encodings(self, apply_scaler: bool = True) -> None:
        for enc in self.__structure.get_encoding_type_strings():
            train = np.array([self.__structure.generate_encoding(enc, self.__train[i], apply_scaler)
                              for i in range(self.__training_size)])
            val = np.array([self.__structure.generate_encoding(enc, self.__val[i], apply_scaler)
                            for i in range(self.__validation_size)])
            test = np.array([self.__structure.generate_encoding(enc, self.__test[i], apply_scaler)
                             for i in range(self.__test_size)])
            self.__data_encoded[enc] = {"training": train, "validation": val, "test": test}

    def generate_ground_truth(self, ground_truth_computers: List[GroundTruthComputer]) -> None:
        for gro in ground_truth_computers:
            train = np.array([gro.compute(self.__train[i])
                              for i in range(self.__training_size)])
            val = np.array([gro.compute(self.__val[i])
                            for i in range(self.__validation_size)])
            test = np.array([gro.compute(self.__test[i])
                             for i in range(self.__test_size)])
            name_gro = gro.get_name()
            self.__data_encoded[name_gro] = {"training": train, "validation": val, "test": test}

    def create_dataset_warm_up_from_encoding_ground_truth(self, n_pairs: int, encoding_type: str, ground_truth: GroundTruthComputer) -> None:
        trees = [self.__structure.generate_tree() for _ in range(n_pairs * 4)]
        encodings = np.array([self.__structure.generate_encoding(encoding_type, t, True) for t in trees])
        grounds = np.array([ground_truth.compute(t) for t in trees])
        data = NumericalData(encodings, grounds)
        X, y = data.get_points_and_labels()
        X, y, _ = PairSampler.random_sampler(X, y, [], n_pairs)
        if encoding_type not in self.__warm_up_data:
            self.__warm_up_data[encoding_type] = {}
        self.__warm_up_data[encoding_type][ground_truth.get_name()] = NumericalData(X, y)

    def create_dataset_warm_up_from_csv(self, file_path: str, file_name: str, train_size: int):
        fey_eq_wu = pd.read_csv(file_path)
        encoding_types = self.__structure.get_encoding_type_strings()
        df = {k: {"left": [], "right": []} for k in encoding_types}
        n_pairs = len(fey_eq_wu)

        for k in df.keys():
            for i in range(n_pairs):
                first_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 0])
                second_formula = tree_from_prefix_repr(fey_eq_wu.iloc[i, 1])
                first_encode = self.__structure.generate_encoding(k, first_formula, True)
                second_encode = self.__structure.generate_encoding(k, second_formula, True)
                df[k]["left"].append(first_encode)
                df[k]["right"].append(second_encode)

            X, y = [], []
            swap = True
            for i in range(n_pairs):
                first_encode, second_encode = df[k]["left"][i], df[k]["right"][i]
                if swap:
                    pair = np.concatenate((first_encode, second_encode), axis=None)
                    y.append(-1)
                else:
                    pair = np.concatenate((second_encode, first_encode), axis=None)
                    y.append(1)
                swap = not swap
                X.append(pair)
            X, y = np.array(X), np.array(y)
            df[k].pop("left")
            df[k].pop("right")
            df[k]["full"] = NumericalData(X, y)
            y_train, y_test = y[:train_size], y[train_size:]
            X_train, X_test = X[:train_size], X[train_size:]
            df[k]["training"] = NumericalData(X_train, y_train)
            df[k]["test"] = NumericalData(X_test, y_test)

        PicklePersist.compress_pickle(self.__folder_path + "/" + file_name, df)
