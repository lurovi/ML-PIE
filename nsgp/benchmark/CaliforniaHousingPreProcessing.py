import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.datasets import fetch_california_housing
from util.PicklePersist import PicklePersist
import random
import numpy as np
pd.options.display.max_columns = 999


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    df = fetch_california_housing(as_frame=True).frame  # change to your local path in which the dataset is located
    print(df.head())
    train, test = train_test_split(df, test_size=0.1, random_state=42)
    train, val = train_test_split(train, test_size=0.2/0.9, random_state=42)

    train_y, val_y, test_y = train["MedHouseVal"].copy(), val["MedHouseVal"].copy(), test["MedHouseVal"].copy()
    train_X, val_X, test_X = train.drop(["MedHouseVal"], axis=1, inplace=False), val.drop(["MedHouseVal"], axis=1, inplace=False), test.drop(["MedHouseVal"], axis=1, inplace=False)

    print(train_X.describe(percentiles=[.25, .50, .75, .85, .95]))

    # scaler = Pipeline([("power", PowerTransformer()), ("minmax", MinMaxScaler())])
    scaler = RobustScaler()
    scaler.fit(train_X)
    #print(pd.DataFrame(scaler.transform(train_X)).describe(percentiles=[.25, .50, .75, .85, .95]))

    X_train, y_train = scaler.transform(train_X), train_y.values
    X_dev, y_dev = scaler.transform(val_X), val_y.values
    X_test, y_test = scaler.transform(test_X), test_y.values

    PicklePersist.compress_pickle("california", {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)})
