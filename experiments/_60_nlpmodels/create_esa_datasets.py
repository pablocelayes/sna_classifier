#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from gensim import matutils
from scipy.sparse import vstack
import pickle, os
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import json
from os.path import join

from experiments._65_linkedarticles.extractors import EsaFeatureExtractor
from tw_dataset.settings import PREFIX, DATAFRAMES_FOLDER, DATASETS_FOLDER
from tw_dataset.dbmodels import *
from multiprocessing import Pool, Manager, Process, log_to_stderr
import logging
mpl = log_to_stderr()
mpl.setLevel(logging.INFO)

tu_path = "/home/pablo/Proyectos/tesiscomp/experiments/_1_one_user_learn_neighbours/active_and_central.json"
TEST_USERS_ALL = json.load(open(tu_path))

USER_ID = 37226353



def _calculate_mean_and_std_deviation(X):
    """
    Calculates mean and standard deviation of sample features.

    Parameters
    ----------
    X : array-like, samples, shape = (n_samples, n_features)
    """

    _, n_features = X.shape

    theta_ = np.zeros((n_features))
    sigma_ = np.zeros((n_features))
    epsilon = 1e-9

    theta_[:] = np.mean(X[:, :], axis=0)
    sigma_[:] = np.std(X[:, :], axis=0) + epsilon

    return theta_, sigma_

def _normalize(X):
    """
    Normalizes sample features.

    self.theta_ and self.sigma_ have to be set.

    Parameters
    ----------
    X : array-like, samples, shape = (n_samples, n_features)
    """
    n_samples, n_features = X.shape
    theta_, sigma_ = _calculate_mean_and_std_deviation(X)   
    new_X = np.zeros(shape=(n_samples, n_features), dtype=np.float32)
    new_X[:, :] = (X[:, :] - theta_[:]) / sigma_[:]

    return new_X

def normalize(X_train, X_test):
    train_size = X_train.shape[0]
    X = vstack((X_train, X_test))
    X = _normalize(X)

    X_train = X[:train_size,:]
    X_test = X[train_size:,:]

    return X_train, X_test

def load_dataframe(uid=USER_ID):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtest_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ys_%d.pickle" % uid)
    exists = False
    if os.path.exists(Xtrain_fname):
        try:
            X_train = pd.read_pickle(Xtrain_fname)
            X_test = pd.read_pickle(Xtest_fname)        
            y_train, y_test = pickle.load(open(ys_fname, 'rb'))
            exists = True
        except Exception as e:
            pass

    return X_train, X_test, y_train, y_test

# Extract features
def load_esa_dataset(uid):
    fname = join(DATASETS_FOLDER, 'esads_%d.npz' % uid)
    z = np.load(open(fname,'rb'))
    X_train_esa = z['arr_0']
    X_test_esa = z['arr_1']
    y_train = z['arr_2']
    y_test = z['arr_3']

    return X_train_esa, X_test_esa, y_train, y_test


if __name__ == '__main__':
    def worker(its, rows_array):
        for i, t in its:
            rows_array[i] = extractor.get_features(text=t.text)
            # if i % 10 == 0:
            #     sofar = len(x for x in rows_array if x)
            #     perc = sofar * 100.0 / len(rows_array)
            #     print ("%.2f %% so far" % perc )

    for uid, _, _ in TEST_USERS_ALL:
        # create_esa_dataset(uid)

    # uid = USER_ID
    
        fname = 'esads_%d.npz' % uid

        extractor = EsaFeatureExtractor(PREFIX)
        s = open_session()
        X_train, X_test, y_train, y_test = load_dataframe(uid)
        
        X_train['y'] = y_train
        X_test['y'] = y_test

        train_size = 700
        if len(X_train) > train_size: 
            X_train = X_train.sample(train_size)
            test_size = (3 * train_size) / 7
            X_test = X_test.sample(test_size)

        y_train = X_train['y']
        y_test = X_test['y']

        train_tweets = [s.query(Tweet).get(twid) for twid in X_train.index]
        test_tweets = [s.query(Tweet).get(twid) for twid in X_test.index]

        manager = Manager()
        rows_train = manager.list([None] * len(train_tweets))
        rows_test = manager.list([None] * len(test_tweets))

        NWORKERS = 7
        p = Pool(NWORKERS)
        enumtweets = list(enumerate(train_tweets))
        batch_size = len(train_tweets) / (NWORKERS - 1)
        for i in range(NWORKERS):
            its = enumtweets[i * batch_size: (i+1) * batch_size]
            p.apply_async(worker, (its, rows_train))

        enumtweets = list(enumerate(test_tweets))
        batch_size = len(test_tweets) / (NWORKERS - 1)
        for i in range(NWORKERS):
            its = enumtweets[i * batch_size: (i+1) * batch_size]
            p.apply_async(worker, (its, rows_test))

        p.close()
        p.join()

        X_train_esa = vstack(list(rows_train))
        X_test_esa = vstack(list(rows_test))

        np.savez(open(fname,'wb'), X_train_esa, X_test_esa, y_train, y_test)
