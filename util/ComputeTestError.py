import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

from exps.SklearnDatasetPreProcesser import SklearnDatasetPreProcessor
from genepro.node import Node
from genepro.util import compute_linear_scaling, tree_from_prefix_repr


def test_individual(tree: Node, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                    linear_scaling=True) -> tuple[float, float]:
    prediction: np.ndarray = np.core.umath.clip(tree(X), -1e+10, 1e+10)
    test_prediction: np.ndarray = np.core.umath.clip(tree(X_test), -1e+10, 1e+10)
    if linear_scaling:
        slope, intercept = compute_linear_scaling(y, prediction)
        slope = np.core.umath.clip(slope, -1e+10, 1e+10)
        intercept = np.core.umath.clip(intercept, -1e+10, 1e+10)
        prediction = intercept + np.core.umath.clip(slope * prediction, -1e+10, 1e+10)
        prediction = np.core.umath.clip(prediction, -1e+10, 1e+10)
        test_prediction = intercept + np.core.umath.clip(slope * test_prediction, -1e+10, 1e+10)
        test_prediction = np.core.umath.clip(test_prediction, -1e+10, 1e+10)
    train_mse: float = np.square(np.core.umath.clip(prediction - y, -1e+20, 1e+20)).sum() / float(
        X.shape[0])
    test_mse: float = np.square(np.core.umath.clip(test_prediction - y_test, -1e+20, 1e+20)).sum() / float(
        X_test.shape[0])
    if train_mse > 1e+20:
        train_mse = 1e+20
    if test_mse > 1e+20:
        test_mse = 1e+20
    return train_mse, test_mse


def test_from_file(filename: str, target_filename: str = None):
    if target_filename is None:
        target_filename = filename.replace(".csv", "_test.csv")
    df = pd.read_csv(filename)
    test_mses = []
    for _, row in df.iterrows():
        dataset_name = row['data']
        split_seed = row['split_seed']
        data = SklearnDatasetPreProcessor.load_data(dataset_name=dataset_name, rng_seed=split_seed)
        parsable_tree = row['parsable_tree']
        tree = tree_from_prefix_repr(parsable_tree)
        training_X = data["training"][0]
        training_y = data["training"][1]
        validation_X = data["validation"][0]
        validation_y = data["validation"][1]
        test_X = data["test"][0]
        test_y = data["test"][1]
        test_X = np.concatenate((validation_X, test_X), axis=0)
        test_y = np.concatenate((validation_y, test_y), axis=None)
        _, test_mse = test_individual(tree, training_X, training_y, test_X, test_y, True)
        test_mses.append(test_mse)
    df["test_mse"] = test_mses
    df.to_csv(target_filename, index=False)


def test_from_folder(folder: str, target_folder: str = None):
    for file in listdir(folder):
        filename = join(folder, file)
        if isfile(filename):
            if target_folder is not None:
                test_from_file(filename, join(target_folder, file))
            else:
                test_from_file(filename)


if __name__ == '__main__':
    base_folder = 'D:\Research\ML-PIE\gp_simulated'
    folders = ['train_results_gp_cross_dataset', 'train_results_gp_simulated_user_constant_rate',
               'train_results_gp_simulated_user_lazy_end', 'train_results_gp_simulated_user_lazy_start']
    for folder in folders:
        target_folder = folder.replace("train", "test")
        test_from_folder(folder, target_folder)
