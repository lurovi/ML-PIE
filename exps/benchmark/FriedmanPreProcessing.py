import random
import numpy as np
from sklearn.datasets import make_friedman1

from util.PicklePersist import PicklePersist

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    X_train, y_train = make_friedman1(n_samples=400, n_features=5, noise=0.25, random_state=42)
    random.seed(43)
    np.random.seed(43)
    X_dev, y_dev = make_friedman1(n_samples=200, n_features=5, noise=0.25, random_state=43)
    random.seed(44)
    np.random.seed(44)
    X_test, y_test = make_friedman1(n_samples=100, n_features=5, noise=0.25, random_state=44)

    PicklePersist.compress_pickle("friedman1", {"training": (X_train, y_train),
                                                 "validation": (X_dev, y_dev),
                                                 "test": (X_test, y_test)})
