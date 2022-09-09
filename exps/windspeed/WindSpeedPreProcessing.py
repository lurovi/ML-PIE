import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, RobustScaler

from util.PicklePersist import PicklePersist

pd.options.display.max_columns = 999


if __name__ == "__main__":
    df = pd.read_csv("wind_dataset.csv")  # change to your local path in which the dataset is located
    df = df.drop(["DATE"], axis=1, inplace=False)
    df = df.dropna(inplace=False)
    df = df.reset_index(drop=True, inplace=False)
    df = df[list(df.columns.difference(["WIND"])) + ["WIND"]]

    train, test = train_test_split(df, test_size=0.1, random_state=1)
    train, val = train_test_split(train, test_size=0.2/0.9, random_state=1)

    train_y, val_y, test_y = train["WIND"].copy(), val["WIND"].copy(), test["WIND"].copy()
    train_X, val_X, test_X = train.drop(["WIND"], axis=1, inplace=False), val.drop(["WIND"], axis=1, inplace=False), test.drop(["WIND"], axis=1, inplace=False)

    print(train_X.describe(percentiles=[.25, .50, .75, .85, .95]))

    # scaler = Pipeline([("power", PowerTransformer()), ("minmax", MinMaxScaler())])
    scaler = RobustScaler()
    scaler.fit(train_X)
    print(pd.DataFrame(scaler.transform(train_X)).describe(percentiles=[.25, .50, .75, .85, .95]))

    X_train, y_train = scaler.transform(train_X), train_y.values
    X_dev, y_dev = scaler.transform(val_X), val_y.values
    X_test, y_test = scaler.transform(test_X), test_y.values

    PicklePersist.compress_pickle("wind_dataset_split", {"training":(X_train, y_train), "validation":(X_dev, y_dev), "test":(X_test, y_test)})
