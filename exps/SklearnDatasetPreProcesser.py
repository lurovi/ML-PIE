import random
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from functools import partial
import warnings


class SklearnDatasetPreProcessor:

    @staticmethod
    def load_data(dataset_name: str, rng_seed: int, previous_seed: int = None, path_dict: Dict[str, str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if path_dict is None:
            path_dict = {}
        switch = {
                 "boston":
                 partial(SklearnDatasetPreProcessor.boston, rng_seed=rng_seed, previous_seed=previous_seed),
                 "heating":
                 partial(SklearnDatasetPreProcessor.heating, rng_seed=rng_seed, previous_seed=previous_seed, path=path_dict.get("heating")),
                 "cooling":
                 partial(SklearnDatasetPreProcessor.cooling, rng_seed=rng_seed, previous_seed=previous_seed, path=path_dict.get("cooling"))
                 }
        return switch.get(dataset_name)()

    @staticmethod
    def boston(rng_seed: int, previous_seed: int = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            data, target = load_boston(return_X_y=True)
        df = pd.DataFrame(data=data)
        df["target"] = target
        #print(df.head())
        #print(df.shape)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        #print(df.shape)
        #print(df.head())
        train, test = train_test_split(df, test_size=0.1, random_state=rng_seed)
        train, val = train_test_split(train, test_size=0.2 / 0.9, random_state=rng_seed)

        train_y, val_y, test_y = train["target"].copy(), val["target"].copy(), test["target"].copy()
        train_X, val_X, test_X = train.drop(["target"], axis=1, inplace=False), val.drop(["target"], axis=1,
                                                                                         inplace=False), test.drop(
            ["target"], axis=1, inplace=False)

        #print(train_X.describe(percentiles=[.25, .50, .75, .85, .95]))

        # scaler = Pipeline([("power", PowerTransformer()), ("minmax", MinMaxScaler())])
        scaler = RobustScaler()
        scaler.fit(train_X)
        # print(pd.DataFrame(scaler.transform(train_X)).describe(percentiles=[.25, .50, .75, .85, .95]))

        X_train, y_train = scaler.transform(train_X), train_y.values
        X_dev, y_dev = scaler.transform(val_X), val_y.values
        X_test, y_test = scaler.transform(test_X), test_y.values

        random.seed(previous_seed)
        np.random.seed(previous_seed)

        return {"training": (X_train, y_train),
                "validation": (X_dev, y_dev),
                "test": (X_test, y_test)}

    @staticmethod
    def heating(rng_seed: int, path: str, previous_seed: int = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        # "benchmark/energyefficiency.xlsx"
        df = pd.read_excel(io=path)
        # Y1: heating
        # Y2: cooling
        # print(df.head())
        train, test = train_test_split(df, test_size=0.1, random_state=rng_seed)
        train, val = train_test_split(train, test_size=0.2 / 0.9, random_state=rng_seed)
        train.dropna(inplace=True)
        train.reset_index(drop=True, inplace=True)
        val.dropna(inplace=True)
        val.reset_index(drop=True, inplace=True)
        test.dropna(inplace=True)
        test.reset_index(drop=True, inplace=True)
        train_y1, val_y1, test_y1 = train["Y1"].copy(), val["Y1"].copy(), test["Y1"].copy()
        train_y2, val_y2, test_y2 = train["Y2"].copy(), val["Y2"].copy(), test["Y2"].copy()

        train_X, val_X, test_X = train.drop(["Y1", "Y2"], axis=1, inplace=False), val.drop(["Y1", "Y2"], axis=1,
                                                                                           inplace=False), test.drop(
            ["Y1", "Y2"], axis=1, inplace=False)

        # print(train_X.describe(percentiles=[.25, .50, .75, .85, .95]))

        # scaler = Pipeline([("power", PowerTransformer()), ("minmax", MinMaxScaler())])
        scaler = RobustScaler()
        scaler.fit(train_X)
        # print(pd.DataFrame(scaler.transform(train_X)).describe(percentiles=[.25, .50, .75, .85, .95]))

        X_train, y_train = scaler.transform(train_X), train_y1.values
        X_dev, y_dev = scaler.transform(val_X), val_y1.values
        X_test, y_test = scaler.transform(test_X), test_y1.values

        random.seed(previous_seed)
        np.random.seed(previous_seed)

        return {"training": (X_train, y_train),
                "validation": (X_dev, y_dev),
                "test": (X_test, y_test)}

    @staticmethod
    def cooling(rng_seed: int, path: str, previous_seed: int = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        # "benchmark/energyefficiency.xlsx"
        df = pd.read_excel(io=path)  # change to your local path in which the dataset is located
        # Y1: heating
        # Y2: cooling
        # print(df.head())
        train, test = train_test_split(df, test_size=0.1, random_state=rng_seed)
        train, val = train_test_split(train, test_size=0.2 / 0.9, random_state=rng_seed)
        train.dropna(inplace=True)
        train.reset_index(drop=True, inplace=True)
        val.dropna(inplace=True)
        val.reset_index(drop=True, inplace=True)
        test.dropna(inplace=True)
        test.reset_index(drop=True, inplace=True)
        train_y1, val_y1, test_y1 = train["Y1"].copy(), val["Y1"].copy(), test["Y1"].copy()
        train_y2, val_y2, test_y2 = train["Y2"].copy(), val["Y2"].copy(), test["Y2"].copy()

        train_X, val_X, test_X = train.drop(["Y1", "Y2"], axis=1, inplace=False), val.drop(["Y1", "Y2"], axis=1,
                                                                                           inplace=False), test.drop(
            ["Y1", "Y2"], axis=1, inplace=False)

        # print(train_X.describe(percentiles=[.25, .50, .75, .85, .95]))

        # scaler = Pipeline([("power", PowerTransformer()), ("minmax", MinMaxScaler())])
        scaler = RobustScaler()
        scaler.fit(train_X)
        # print(pd.DataFrame(scaler.transform(train_X)).describe(percentiles=[.25, .50, .75, .85, .95]))

        X_train, y_train = scaler.transform(train_X), train_y1.values
        X_dev, y_dev = scaler.transform(val_X), val_y1.values
        X_test, y_test = scaler.transform(test_X), test_y1.values

        y_train = train_y2.values
        y_dev = val_y2.values
        y_test = test_y2.values

        random.seed(previous_seed)
        np.random.seed(previous_seed)

        return {"training": (X_train, y_train),
                "validation": (X_dev, y_dev),
                "test": (X_test, y_test)}

