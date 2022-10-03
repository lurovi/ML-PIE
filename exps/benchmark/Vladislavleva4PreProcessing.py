import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.datasets import fetch_california_housing
from util.PicklePersist import PicklePersist
import random
pd.options.display.max_columns = 999


def target(x: np.ndarray):
    s = 0.0
    for i in range(5):
        s += (x[i] - 3.0) ** 2
    return 10.0/float(5.0 + s)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    X_train = np.random.uniform(0.05, 6.05 + 1e-4, (1024, 7))
    X_dev = np.random.uniform(-0.25, 6.35 + 1e-4, (50, 7))
    X_test = np.random.uniform(-0.25, 6.35 + 1e-4, (30, 7))
    print(X_train[0])
    print(X_train[1])
    y_train = np.array([target(X_train[i]) for i in range(X_train.shape[0])])
    y_dev = np.array([target(X_dev[i]) for i in range(X_dev.shape[0])])
    y_test = np.array([target(X_test[i]) for i in range(X_test.shape[0])])
    print(y_train[0])
    print(y_train[1])
    PicklePersist.compress_pickle("vladislavleva", {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)})
