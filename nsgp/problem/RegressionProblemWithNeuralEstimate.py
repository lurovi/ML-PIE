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
        self.__linear_scaling = linear_scaling
        self.__uncertainties = []

    def _evaluate(self, x, out, *args, **kwargs):
        if self.mutex is not None:
            with self.mutex:
                self._eval(x, out)
        else:
            self._eval(x, out)

    def _eval(self, x, out):
        out["F"] = np.empty((len(x), 2), dtype=np.float32)
        current_uncertainties = []
        for i in range(len(x)):
            tree = x[i, 0]
            prediction: np.ndarray = np.core.umath.clip(tree(self.__X), -1e+10, 1e+10)
            slope, intercept = 1.0, 0.0
            if self.__linear_scaling:
                slope, intercept = compute_linear_scaling(self.__y, prediction)
                slope = np.core.umath.clip(slope, -1e+10, 1e+10)
                intercept = np.core.umath.clip(intercept, -1e+10, 1e+10)
                prediction = intercept + np.core.umath.clip(slope * prediction, -1e+10, 1e+10)
                prediction = np.core.umath.clip(prediction, -1e+10, 1e+10)
            mse: float = np.square(np.core.umath.clip(prediction - self.__y, -1e+20, 1e+20)).sum() / float(
                self.__n_records)
            if mse > 1e+20:
                mse = 1e+20
            out["F"][i, 0] = mse
            encoded_tree = self.__tree_encoder.encode(tree, True)
            tensor_encoded_tree = torch.from_numpy(encoded_tree).float().reshape(1, -1)
            interpretability, uncertainty, _ = self.__interpretability_estimator.predict(tensor_encoded_tree)
            out["F"][i, 1] = -1 * interpretability[0][0].item()
            current_uncertainties.append(uncertainty[0])
        self.__uncertainties.append(current_uncertainties)

    def get_uncertainties(self):
        return self.__uncertainties

    def reset_uncertainties(self):
        self.__uncertainties = []

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle mutex
        del state["mutex"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mutex = None
