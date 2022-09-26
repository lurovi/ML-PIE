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
    val = x[0]
    s = 0.0
    for i in range(1, val+1):
        s += 1.0/float(i)
    return s


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    X_train = np.array([i for i in range(1, 50 + 1)]).reshape(-1, 1)
    X_dev = np.array([i for i in range(1, 120 + 1)]).reshape(-1, 1)
    X_test = np.array([i for i in range(1, 240 + 1)]).reshape(-1, 1)
    print(X_train[0])
    print(X_train[1])
    y_train = np.array([target(X_train[i]) for i in range(X_train.shape[0])])
    y_dev = np.array([target(X_dev[i]) for i in range(X_dev.shape[0])])
    y_test = np.array([target(X_test[i]) for i in range(X_test.shape[0])])
    print(y_train[0])
    print(y_train[1])
    PicklePersist.compress_pickle("keijzer", {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)})
