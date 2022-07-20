import pickle
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.utils.multiclass import type_of_target
from sklearn import metrics
import random
from deeplearn.neuralnet import spearman_footrule_direct
from util.sort import heapsort


# ==============================================================================================================
# DATASET HANDLER
# ==============================================================================================================


def build_numpy_dataset_twopointscompare(X, y, num_points_to_generate, binary_label=False):
    original_number_of_points = len(X)  # the number of points in dataset instance
    new_data = []
    new_labels = []
    indexes = list(range(original_number_of_points))  # [0, 1, 2, ..., original_number_of_points-2, original_number_of_points-1]
    for _ in range(num_points_to_generate):
        idx = random.choices(indexes, k=2)  # extract two points at random with replacement (for computational efficiency reasons)
        first_point = X[idx[0]]  # first point extracted
        first_label = y[idx[0]]  # first label extracted
        second_point = X[idx[1]]  # second point extracted
        second_label = y[idx[1]]  # second label extracted
        if first_label >= second_label:  # first point has a higher score than the second one
            if binary_label:
                new_labels.append(1)  # close to one when the first point is higher: sigmoid(z_final) >= 0.5
            else:
                new_labels.append(-1.0)  # if the first point is higher, then the loss decreases: -1*(p1-p2)
        else:  # first point has a lower score than the second one
            if binary_label:
                new_labels.append(0)  # close to zero when the first point is lower: sigmoid(z_final) < 0.5
            else:
                new_labels.append(1.0)  # if the second point is higher, then the loss decreases: 1*(p1-p2)
        new_data.append(first_point.tolist() + second_point.tolist())
    return np.array(new_data), np.array(new_labels)


# ==============================================================================================================
# MODEL PERFORMANCE
# ==============================================================================================================


def model_accuracy(confusion_matrix):
    N = confusion_matrix.shape[0]
    acc = sum([confusion_matrix[i, i] for i in range(N)])
    class_performance = {i: {} for i in range(N)}
    for i in range(N):
        positive_rate = confusion_matrix[i, i]
        true_positives = confusion_matrix[i, :].sum()
        predicted_positives = confusion_matrix[:, i].sum()
        class_performance[i]["precision"] = positive_rate / predicted_positives if predicted_positives != 0 else 0
        class_performance[i]["recall"] = positive_rate / true_positives if true_positives != 0 else 0
        class_performance[i]["f1"] = (2 * class_performance[i]["precision"] * class_performance[i]["recall"]) / (
                    class_performance[i]["precision"] + class_performance[i]["recall"]) if class_performance[i][
                                                                                               "precision"] + \
                                                                                           class_performance[i][
                                                                                               "recall"] != 0 else 0
    return acc / confusion_matrix.sum(), class_performance


def compute_binary_confusion_matrix_metrics(y_true, y_pred):
    """
    This method gets as input an array of ground-truth labels (y_true) and an array of predicted labels (y_pred).
    It outputs a dictionary with confusion matrix values and common evaluation metrics.
    This method works for binary classification problems where the positive class is depicted with 1 while negative class is depicted with 0.
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)  # sensitivity
    specificity = tn/(tn+fp)
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    f1 = (2*precision*recall)/(precision+recall)
    mcc = ( (tn*tp) - (fp*fn) )/np.sqrt( (tn+fn)*(fp+tp)*(tn+fp)*(fn+tp) )
    roc_auc = metrics.roc_auc_score(y_true, y_pred)
    return {"total number of predictions": len(y_true), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": accuracy, "f1": f1, "mcc": mcc, "precision": precision, "recall": recall,
            "specificity": specificity, "fp_rate": fp_rate, "fn_rate": fn_rate, "roc_auc_score": roc_auc}


def model_cost(confusion_matrix, cost_matrix):
    N = confusion_matrix.shape[0]
    a = np.subtract(cost_matrix, np.identity(N))
    return np.multiply(confusion_matrix, a).sum()


def evaluate_ml_ranking_with_spearman_footrule(X, y, ml_estimator):
    y_true = np.argsort(y, kind="heapsort")[::-1]
    comparator = partial(ml_comparator, ml_estimator=ml_estimator)
    _, y_pred = heapsort([X[i] for i in range(len(X))], comparator, inplace=False, reverse=True)
    return spearman_footrule_direct(y_true, np.array(y_pred))


# ==============================================================================================================
# COMPARATOR
# ==============================================================================================================


def ml_comparator(point_1, point_2, ml_estimator):
    point = np.concatenate((point_1, point_2), axis=None).reshape(1, -1)
    res = ml_estimator.estimate(point)[0]
    return res == 0


# ==============================================================================================================
# ESTIMATOR MODELS
# ==============================================================================================================


class Estimator(ABC):
    def __init__(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model

    @abstractmethod
    def train(self, X, y, save_path=None, verbose=False):
        pass

    @abstractmethod
    def estimate(self, X):
        pass


class MLEstimator(Estimator):
    def __init__(self, model, space, scoring, random_state, n_splits=5, n_repeats=3, n_jobs=-1,
                 randomized_search=False, n_iter=800):
        super(MLEstimator, self).__init__(model)
        self.space = space
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.randomized_search = randomized_search
        self.n_iter = n_iter

    def train(self, X, y, save_path=None, verbose=False):
        if not (type_of_target(y) == 'continuous'):
            cv = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats,
                                         random_state=self.random_state)
        else:
            cv = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        refit = True if isinstance(self.scoring, str) else self.scoring[0]
        if self.randomized_search:
            search = RandomizedSearchCV(self.get_model(), self.space, scoring=self.scoring, n_jobs=self.n_jobs, cv=cv,
                                        random_state=self.random_state, n_iter=self.n_iter, refit=refit)
        else:
            search = GridSearchCV(self.get_model(), self.space, scoring=self.scoring, n_jobs=self.n_jobs,
                                  cv=cv, refit=refit)
        search.fit(X, y)
        if verbose:
            print('Best Estimator: %s' % search.best_estimator_)
            print('Best Score: %s' % search.best_score_)
            print('Best Hyper-parameters: %s' % search.best_params_)
            print("=" * 50)
        self.set_model(search)
        if not (save_path is None):
            pickling_on = open(save_path, "wb")
            pickle.dump(self, pickling_on)
            pickling_on.close()

    def estimate(self, X):
        return self.get_model().predict(X)


