import random
import numpy as np
from sklearn.datasets import make_friedman1

from util.PicklePersist import PicklePersist

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    X_train, y_train = make_friedman1(n_samples=700, n_features=7, noise=0.25, random_state=42)
    X_dev, y_dev = make_friedman1(n_samples=400, n_features=7, noise=0.25, random_state=42)
    X_test, y_test = make_friedman1(n_samples=200, n_features=7, noise=0.25, random_state=42)

    PicklePersist.compress_pickle("friedman1", {"training": (X_train, y_train),
                                                 "validation": (X_dev, y_dev),
                                                 "test": (X_test, y_test)})
