import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.datasets import load_diabetes
from util.PicklePersist import PicklePersist
import random
import numpy as np
pd.options.display.max_columns = 999


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    df = load_diabetes(as_frame=True).frame  # change to your local path in which the dataset is located
    print(df.head())
    # Sex 1 is negative (< 0) while sex 2 is positive (> 0)
    df["sex1"] = np.where(df["sex"] < 0, 1, 0)
    df["sex2"] = np.where(df["sex"] >= 0, 1, 0)
    df.drop(["sex"], axis=1, inplace=True)
    print(df.shape)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.shape)
    print(df.head())
    train, test = train_test_split(df, test_size=0.1, random_state=42)
    train, val = train_test_split(train, test_size=0.2/0.9, random_state=42)

    train_y, val_y, test_y = train["target"].copy(), val["target"].copy(), test["target"].copy()
    train_X, val_X, test_X = train.drop(["target"], axis=1, inplace=False), val.drop(["target"], axis=1, inplace=False), test.drop(["target"], axis=1, inplace=False)

    print(train_X.describe(percentiles=[.25, .50, .75, .85, .95]))

    X_train, y_train = train_X.values, train_y.values
    X_dev, y_dev = val_X.values, val_y.values
    X_test, y_test = test_X.values, test_y.values

    PicklePersist.compress_pickle("diabets", {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)})
