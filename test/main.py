# ========================================================
# COMMON IMPORTS
# ========================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import bz2
import pickle
import _pickle as cPickle
import random
import os
import io
import time
import copy
import json
import string
import copy
import argparse
import sys
import math

from abc import ABC, abstractmethod
from PIL import Image, ImageTk
from functools import partial

from numpy.random import default_rng

from threading import Semaphore, Lock, RLock, Thread

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
from sklearn.model_selection import learning_curve, permutation_test_score, validation_curve
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.utils.multiclass import type_of_target

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler


# ========================================================
# COMMON SETTINGS
# ========================================================

# Hrule--> "====="
hrule = lambda x: "="*x
Hrule = lambda x, y: "="*(x//2)+y+"="*(x//2)

sns.set(style="whitegrid") #White Grid
sns.set(rc={'figure.figsize':(10,8)})
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
pd.options.display.float_format = '{:.3f}'.format


# ========================================================
# COMMON METHODS
# ========================================================


# Saves the "data" with the "title" and adds the .pkl
# title is the path of the filename to create without extension, data is what you want to make persistent
# note that this method creates a new file from scratch, but it does not create new folders
# if you want to save this file into a directory that is different from the current directory, please make sure that
# all directories in the path have already been created in the correct order given by the hierarchical structure
# of the input path.
def full_pickle(title, data):
    pikd = open(title + '.pkl', 'wb')
    pickle.dump(data, pikd)
    pikd.close()


# loads and returns a pickled objects
# file is the path of the file to load, extension included
def loosen(file):
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data


# Pickle a file and then compress it into a file with extension
# title is the path of the filename to create without extension, data is what you want to make persistent
# note that this method creates a new file from scratch, but it does not create new folders
# if you want to save this file into a directory that is different from the current directory, please make sure that
# all directories in the path have already been created in the correct order given by the hierarchical structure
# of the input path.
def compress_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


# Load any compressed pickle file
# file is the path of the file to load, extension included
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    return cPickle.load(data)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ========================================================
#  UTILITY METHODS
# ========================================================


def softmax_stable(x):
    return np.exp(x - np.max(x))/np.exp(x - np.max(x)).sum()


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = '%s%s' % (n, '%')
    return percentile_


def percentile_multi_axis(n, axis=0):
    def percentile_(x, axis):
        return np.percentile(x, n, axis)
    percentile_.__name__ = '%s%s' % (n, '%')
    return percentile_


# ========================================================
# RANDOM SEED AND DEVICE
# ========================================================

# Setting random seed to allow scientific reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
generator_data_loader = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
generator_data_loader.manual_seed(seed)
torch.use_deterministic_algorithms(True)
# Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================================
# DATASET AND NEURAL NETWORK
# ========================================================


class NumericalData(Dataset):
    def __init__(self, data, transform=None):
        # data is a numpy matrix with N rows and D columns that represents a generic numerical dataset.
        # transform is a sklearn pipeline that scales and transforms the data (e.g., StandardScaler).
        # the transform pipeline must have already been fitted to the training data when this constructor is called.
        self.data = data
        self.transform = transform

    def __len__(self):
        # gets the number of rows in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # gets a data point from the dataset as torch tensor array
        point = self.data[idx]
        if self.transform:
            point = self.transform.transform(point.reshape(1, -1))
        point = torch.from_numpy(point)
        return point.float()


class Autoencoder(nn.Module):  # dim-20-10-7-10-20-dim
    def __init__(self, activation, dim):
        # dim is the number of features of the input dataset
        # activation is the activation function (e.g., nn.ReLU(), nn.Tanh())
        super(Autoencoder, self).__init__()

        self.activation = activation
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(self.dim, 20),
            self.activation,
            nn.Linear(20, 10),
            self.activation,
            nn.Dropout(0.15),
            nn.Linear(10, 7),
            self.activation
        )

        self.decoder = nn.Sequential(
            nn.Linear(7, 10),
            self.activation,
            nn.Dropout(0.15),
            nn.Linear(10, 20),
            self.activation,
            nn.Linear(20, self.dim),
            self.activation
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z


# ========================================================
# TRAINING AND PREDICTION
# ========================================================


def train(dataloader, model, device, optimizer, loss_fn, max_epochs=20, verbose=True):
    loss_arr = []
    loss_epoch_arr = []
    for epoch in range(max_epochs):
        for i, batch in enumerate(dataloader, 0):
            model.train()
            inputs = batch
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())
        loss_epoch_arr.append(loss.item())
        if verbose:
            print(f"Epoch {epoch + 1}/{max_epochs}. Loss: {loss.item()}.")
    return model, loss_arr, loss_epoch_arr


def predict(model, data, loss_fn, device):
    squared_errors = []
    model.eval()
    for i in range(len(data)):
        inputs = data[i]
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, inputs)
        squared_errors.append(loss.item())
    return squared_errors


def train_autoencoder(dataloader, activation, dim, device, learning_rate=0.001, weight_decay=0.0001,
                      max_epochs=20, verbose=True):
    model = Autoencoder(activation=activation, dim=dim)
    model = model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return train(dataloader, model, device, optimizer, criterion, max_epochs=max_epochs, verbose=verbose)


def predict_with_autoencoder(model, data, device):
    criterion = nn.MSELoss(reduction='mean')
    return predict(model, data, criterion, device)


# ========================================================
# MAIN
# ========================================================


if __name__ == '__main__':





    # Initializing data loader with training data
    dataloader = DataLoader(data, batch_size=100, shuffle=True,
                            worker_init_fn=seed_worker, generator=generator_data_loader)

    # Training
    model, loss_arr, loss_epoch_arr = train_autoencoder(dataloader, nn.Tanh(), 13, device)

    # Prediction of normal data
    print(Hrule(30, "Squared errors for first 10 elements of normal data."))
    print(predict_with_autoencoder(model, new_data, device)[:10])

    # Prediction of anomalous data
    print(Hrule(30, "Squared errors for anomalous data."))
    print(predict_with_autoencoder(model, anomalies, device))


