import threading

import numpy as np
import torch
from pymoo.core.problem import Problem

from deeplearn.trainer.Trainer import Trainer
from genepro.util import compute_linear_scaling
from nsgp.encoder.TreeEncoder import TreeEncoder


class RegressionProblemWithNeuralEstimate(Problem):
    def __init__(self, X: np.ndarray, y: np.ndarray, mutex: threading.Lock = None, tree_encoder: TreeEncoder = None,
                 interpretability_estimator: Trainer = None, linear_scaling: bool = True):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, n_eq_constr=0)
        if y.shape[0] != X.shape[0]:
            raise AttributeError(
                f"Number of observations in X ({X.shape[0]}) and y ({y.shape[0]}) mismatched.")
        if len(y.shape) != 1:
            raise AttributeError(
                f"y must be one-dimensional, but here has {len(y.shape)} dimensions.")
        self.__n_records: int = X.shape[0]
        self.__X: np.ndarray = X
        self.__y: np.ndarray = y
        self.mutex = mutex
        self.__tree_encoder = tree_encoder
        self.__interpretability_estimator = interpretability_estimator
        self.linear_scaling = linear_scaling

    def _evaluate(self, x, out, *args, **kwargs):
        if self.mutex is not None:
            with self.mutex:
                self._eval(x, out)
        else:
            self._eval(x, out)

    def _eval(self, x, out):
        out["F"] = np.empty((len(x), 2), dtype=np.float32)
        for i in range(len(x)):
            tree = x[i, 0]
            prediction: np.ndarray = tree(self.__X)
            if self.linear_scaling:
                slope, intercept = compute_linear_scaling(self.__y, prediction)
                prediction = intercept + slope * prediction
            mse: float = np.square(np.clip(prediction - self.__y, -1e+20, 1e+20)).sum() / float(self.__n_records)
            if mse > 1e+20:
                mse = 1e+20
            out["F"][i, 0] = mse
            # TODO this might be done in batch to speed up
            encoded_tree = self.__tree_encoder.encode(tree, True)
            tensor_encoded_tree = torch.from_numpy(encoded_tree).float().reshape(1, -1)
            out["F"][i, 1] = -1 * self.__interpretability_estimator.predict(tensor_encoded_tree)[0][0].item()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle mutex
        del state["mutex"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mutex = None
