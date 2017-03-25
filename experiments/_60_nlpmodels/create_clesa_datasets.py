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

from experiments._65_linkedarticles.extractors import CLEsaFeatureExtractor
from tw_dataset.settings import DATAFRAMES_FOLDER, DATASETS_FOLDER
from tw_dataset.dbmodels import *
from multiprocessing import Pool, Manager, Process, log_to_stderr, Lock
import logging
from random import sample
from scipy.sparse import csc

mpl = log_to_stderr()
mpl.setLevel(logging.ERROR)

PREFIX = '/home/pablo/Proyectos/cogfor/repo/data/'

tu_path = "/home/pablo/Proyectos/tesiscomp/experiments/_1_one_user_learn_neighbours/active_and_central.json"
TEST_USERS_ALL = json.load(open(tu_path))

USER_ID = 37226353

def load_low_recall_users():
    with open('../_1_one_user_learn_neighbours/recalls_test_amb.json') as f:
        # Estas recalls están calculadas sólo para usuarios con <90% recall en el training set
        # en el caso de features sociales
        recalls = json.load(f)
    
    low_recall_users = sorted(recalls.items(), key=lambda (u,r):r)

    return low_recall_users

def sub_sample_negs_df(X, y, neg_to_pos_rate=5):
    npos = int(sum(y))
    neg_inds = [i for i in range(len(y)) if y[i] == 0]
    pos_inds = [i for i in range(len(y)) if y[i]]
    sample_neg_inds = sample(neg_inds, npos * neg_to_pos_rate)
    inds = pos_inds + sample_neg_inds

    Xs = X.iloc[inds,:]
    ys = y[inds]

    return Xs, ys

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

def load_dataframe(uid=USER_ID, small=False):
    if small:
        Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_small_%d.pickle" % uid)
        Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtest_small_%d.pickle" % uid)
        ys_fname = join(DATAFRAMES_FOLDER, "ys_small_%d.pickle" % uid)
    else:
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
def load_clesa_dataset_big(uid, train_size):
    fname = 'clesads_%d' % uid
    if train_size:
        fname += '_tr%d' % train_size
    fname = join(DATASETS_FOLDER, '%s.npz' % fname)
    z = np.load(open(fname,'rb'))
    X_train_clesa = z['arr_0']
    X_test_clesa = z['arr_1']
    y_train = z['arr_2']
    y_test = z['arr_3']

    X_train_clesa = csc.csc_matrix(X_train_clesa.tolist())
    X_test_clesa = csc.csc_matrix(X_test_clesa.tolist())

    return X_train_clesa, X_test_clesa, y_train, y_test


def load_clesa_dataset_small(uid, neg_to_pos_rate):
    fname = 'clesads_small%d_%d' % (neg_to_pos_rate, uid)
    fname = join(DATASETS_FOLDER, '%s.npz' % fname)
    z = np.load(open(fname,'rb'))
    X_train_clesa = z['arr_0']
    X_test_clesa = z['arr_1']
    y_train = z['arr_2']
    y_test = z['arr_3']

    X_train_clesa = csc.csc_matrix(X_train_clesa.tolist())
    X_test_clesa = csc.csc_matrix(X_test_clesa.tolist())

    return X_train_clesa, X_test_clesa, y_train, y_test


def sample_dataset(X_train, X_test, y_train, y_test, train_size):
    X_train['y'] = y_train
    X_test['y'] = y_test

    if len(X_train) > train_size: 
        X_train = X_train.sample(train_size)
        test_size = (3 * train_size) / 7
        X_test = X_test.sample(test_size)

    y_train = X_train['y']
    y_test = X_test['y']

    return X_train, X_test, y_train, y_test

def save_dataset(dataset, fname):
    X_train, X_test, y_train, y_test = dataset
    np.savez(open(fname,'wb'), X_train, X_test, y_train, y_test)

def load_dataset(fname):
    z = np.load(open(fname,'rb'))
    X_train = z['arr_0']
    X_test = z['arr_1']
    y_train = z['arr_2']
    y_test = z['arr_3']

    X_train = csc.csc_matrix(X_train.tolist())
    X_test = csc.csc_matrix(X_test.tolist())

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    low_recall_users = load_low_recall_users()
    # for uid, _, _ in TEST_USERS_ALL:
    for uid, _ in low_recall_users[:1]:
        for neg_to_pos_rate in [4]:
        # for uid, _, _ in [(37226353, 0, 0)]:
            # save = False
            save = True
            neg_to_pos_rate = 4
            def worker(its, rows_array, lock):
                extractor = CLEsaFeatureExtractor(PREFIX, target_prefix='small2k_')
                for i, t in its:
                    feats = extractor.get_features(text=t.text, lang='es')
                    lock.acquire()
                    rows_array[i] = feats
                    lock.release()

                    if i % 10 == 0:
                        sofar = len([
                            x for x in list(rows_array) if x is not None])
                        perc = sofar * 100.0 / len(list(rows_array))
                        print ("%.2f %% so far" % perc )

            # train_size = 70000
            
            # fname = 'clesads_%d.npz' % uid
            # fname = 'clesads_%d' % uid
            # if train_size:
            #     fname += '_tr%d' % train_size
            uid = int(uid)

            s = open_session()
            X_train, X_test, y_train, y_test = load_dataframe(uid)
            
            # Sample train/test
            # X_train, X_test, y_train, y_test = sample_dataset(X_train, X_test, y_train, y_test, train_size)
            X_train, y_train = sub_sample_negs_df(X_train, y_train, neg_to_pos_rate)
            X_test, y_test = sub_sample_negs_df(X_test, y_test, neg_to_pos_rate)

            # Save sample of social features for future comparison
            if save:
                Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_small%d_%d.pickle" % (neg_to_pos_rate, uid))
                Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtest_small%d_%d.pickle" % (neg_to_pos_rate, uid))
                ys_fname = join(DATAFRAMES_FOLDER, "ys_small%d_%d.pickle" % (neg_to_pos_rate, uid))

                X_train.to_pickle(Xtrain_fname)
                X_test.to_pickle(Xtest_fname)
                pickle.dump((y_train, y_test), open(ys_fname, 'wb'))
            
            train_tweets = [s.query(Tweet).get(twid) for twid in X_train.index]
            test_tweets = [s.query(Tweet).get(twid) for twid in X_test.index]

            manager = Manager()
            rows_train = manager.list([None] * len(train_tweets))
            rows_test = manager.list([None] * len(test_tweets))
            lock = manager.Lock()

            NWORKERS = 6
            p = Pool(NWORKERS)
            enumtweets = list(enumerate(train_tweets))
            batch_size = len(train_tweets) / (NWORKERS - 1)
            for i in range(NWORKERS):
                its = enumtweets[i * batch_size: (i+1) * batch_size]
                # worker(its, rows_train, lock)
                p.apply_async(worker, (its, rows_train, lock))

            enumtweets = list(enumerate(test_tweets))
            batch_size = len(test_tweets) / (NWORKERS - 1)
            for i in range(NWORKERS):
                its = enumtweets[i * batch_size: (i+1) * batch_size]
                # worker(its, rows_test, lock)
                p.apply_async(worker, (its, rows_test, lock))

            p.close()
            p.join()

            import ipdb; ipdb.set_trace()
            X_train_clesa = vstack(list(rows_train))
            X_test_clesa = vstack(list(rows_test))
            
            if save:
                fname = 'clesads_small%d_%d' % (neg_to_pos_rate, uid)
                fname = join(DATASETS_FOLDER, '%s.npz' % fname)
                np.savez(open(fname,'wb'), X_train_clesa, X_test_clesa, y_train, y_test)
