#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from __future__ import print_function

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
from os.path import join, exists
from random import sample

from experiments._65_linkedarticles.extractors import LdaFeatureExtractor
from tw_dataset.settings import DATAFRAMES_FOLDER, DATASETS_FOLDER
from tw_dataset.dbmodels import *
from multiprocessing import Pool, Manager, Process, log_to_stderr, Lock
import logging
from random import sample
from scipy.sparse import csc, csc_matrix
from scipy import sparse as sp
from math import ceil
from time import time
import re
from gensim.models import LdaModel
from gensim import corpora
from tokenizer import tokenize, spanish_stopwords


mpl = log_to_stderr()
mpl.setLevel(logging.ERROR)

PREFIX = '/home/pablo/Proyectos/cogfor/repo/data/es'

tu_path = "/home/pablo/Proyectos/tesiscomp/experiments/_1_one_user_learn_neighbours/active_and_central.json"
TEST_USERS_ALL = json.load(open(tu_path))

USER_ID = 37226353

def load_all_f1s():
    with open('../_1_one_user_learn_neighbours/f1s_valid_svc.json') as f:
        f1s = json.load(f)
    
    f1s = f1s.items()
    f1s = sorted(f1s, key=lambda (u,f):f)

    return f1s

def load_nlp_selected_users():
    if not exists('nlp_users.json'):
        with open('../_1_one_user_learn_neighbours/f1s_valid_svc.json') as f:
            f1s = json.load(f)
        
        ur = f1s.items()

        # Nos quedamos con los de valid f1 < 75% ( es un proceso pesado )
        threshold = 0.75
        low_ur = [(u,r) for (u,r) in ur if r < threshold]

        high_ur = [(u,r) for (u,r) in ur if r >= threshold and r < 1]

        # we add a sample of the rest
        res = low_ur + sample(high_ur, 10)

        res = sorted(res, key=lambda (u,r):r)

        with open('nlp_users.json', 'w') as f:
            json.dump(res, f)

    with open('nlp_users.json') as f:
        res = json.load(f)

    return res

def extract_ds_inds(uids):
    for uid in uids:
        print uid
        X_train, X_valid, X_test, y_train, y_valid, y_test = load_validation_dataframe(uid)
        with open('xinds/X_train_inds_%d.json' % uid, 'w') as f:
            json.dump(list(X_train.index), f)
        with open('xinds/X_valid_inds_%d.json' % uid, 'w') as f:
            json.dump(list(X_valid.index), f)
        with open('xinds/X_test_inds_%d.json' % uid, 'w') as f:
            json.dump(list(X_test.index), f)


def load_ds_inds(uid):
    with open('xinds/X_train_inds_%d.json' % uid) as f:
        X_train_inds = json.load(f)
    with open('xinds/X_valid_inds_%d.json' % uid) as f:
        X_valid_inds = json.load(f)
    with open('xinds/X_test_inds_%d.json' % uid) as f:
        X_test_inds = json.load(f)

    return X_train_inds, X_valid_inds, X_test_inds


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

def load_validation_dataframe(uid):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ysv_%d.pickle" % uid)

    X_train = pd.read_pickle(Xtrain_fname)
    X_valid = pd.read_pickle(Xvalid_fname)
    X_test = pd.read_pickle(Xtest_fname)        
    y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

    return X_train, X_valid, X_test, y_train, y_valid, y_test

# Extract features
def load_lda_dataset_big(uid, train_size):
    fname = 'ldads_%d' % uid
    if train_size:
        fname += '_tr%d' % train_size
    fname = join(DATASETS_FOLDER, '%s.npz' % fname)
    z = np.load(open(fname,'rb'))
    X_train_lda = z['arr_0']
    X_test_lda = z['arr_1']
    y_train = z['arr_2']
    y_test = z['arr_3']

    X_train_lda = csc.csc_matrix(X_train_lda.tolist())
    X_test_lda = csc.csc_matrix(X_test_lda.tolist())

    return X_train_lda, X_test_lda, y_train, y_test


def load_lda_dataset_small(uid, neg_to_pos_rate):
    fname = 'ldads_small%d_%d' % (neg_to_pos_rate, uid)
    fname = join(DATASETS_FOLDER, '%s.npz' % fname)
    z = np.load(open(fname,'rb'))
    X_train_lda = z['arr_0']
    X_test_lda = z['arr_1']
    y_train = z['arr_2']
    y_test = z['arr_3']

    X_train_lda = csc.csc_matrix(X_train_lda.tolist())
    X_test_lda = csc.csc_matrix(X_test_lda.tolist())

    return X_train_lda, X_test_lda, y_train, y_test


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


def load_lda_dataset(uid):
    fname = join(DATASETS_FOLDER, 'es_twlda100ds_%d.npz' % uid)

    z = np.load(open(fname,'rb'))
    X_train = z['arr_0'].item()
    X_valid = z['arr_1'].item()
    X_test = z['arr_2'].item()

    # X_train = csc.csc_matrix(X_train.tolist())
    # X_valid = csc.csc_matrix(X_train.tolist())
    # X_test = csc.csc_matrix(X_test.tolist())

    cols_train = X_train.shape[1]
    cols_valid = X_valid.shape[1]
    cols_test = X_test.shape[1]

    maxcols = max(cols_train, cols_valid, cols_test)

    if cols_train < maxcols:
        missing_cols = csc_matrix((X_train.shape[0], maxcols - cols_train), dtype=np.float64)
        X_train = sp.hstack((X_train, missing_cols))

    if cols_valid < maxcols:
        missing_cols = csc_matrix((X_valid.shape[0], maxcols - cols_valid), dtype=np.float64)
        X_valid = sp.hstack((X_valid, missing_cols))

    if cols_test < maxcols:
        missing_cols = csc_matrix((X_test.shape[0], maxcols - cols_test), dtype=np.float64)
        X_test = sp.hstack((X_test, missing_cols))

    ys_fname = join(DATAFRAMES_FOLDER, "ysv_%d_small.pickle" % uid)
    y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def load_small_validation_dataframe(uid):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d_small.pickle" % uid)
    Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d_small.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d_small.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ysv_%d_small.pickle" % uid)

    X_train = pd.read_pickle(Xtrain_fname)
    X_valid = pd.read_pickle(Xvalid_fname)
    X_test = pd.read_pickle(Xtest_fname)        
    y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def preprocess(doc):
    pre_doc = doc
        
    # remove URLs
    pre_doc = re.sub(
        r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ", pre_doc)
        
    # find and split hashtags
    # very simple splitting (TODO: come up with something wittier)
    # split on capital letters, but only if hashtag longer than 5
    # → conditional is to avoid splitting abbr. like "IoT" or "NSA"
    pre_doc = re.sub(r"(?:^|\s)[＃#]{1}(\w+)", 
            lambda s: re.sub(r"([A-Z])", r" \1", s.group(0)) if len(s.group(0)) > 5 else s.group(0), 
            pre_doc)
    pre_doc = re.sub(r"＃|#", " ", pre_doc)
    
    # lowercase everything
    pre_doc = pre_doc.lower()
        
    # remove bullshit
    pre_doc = re.sub(r"\@|\'|\"|\\|…|\/|\-|\||\(|\)|\.|\,|\!|\?|\:|\;|“|”|’|—", " ", pre_doc)
    
    # normalize whitespaces
    pre_doc = re.sub(r"\s+", " ", pre_doc)
    pre_doc = re.sub(r"(^\s)|(\s$)", "", pre_doc)
    
    return pre_doc


if __name__ == '__main__':
    def worker(its, rows_array, lock, counter, n):
        model = LdaModel.load('tweets_es_40it.lda')
        dictionary = corpora.Dictionary.load('tweets_es.dict')
        for i, t in its:
            tokens = tokenize(preprocess(t.text))
            doc_bow = dictionary.doc2bow(tokens)
            doc_lda = model[doc_bow]

            feats = doc_lda
            lock.acquire()
            rows_array[i] = feats
            counter.value += 1
            if i % 50 == 0:
                sofar = counter.value
                perc = sofar * 100.0 / n
                print ("%.2f %% so far" % perc )
            lock.release()

    f1s = load_nlp_selected_users()
    # f1s = load_all_f1s()

    start = False

    for uid, r in f1s:
    # for uid, r in f1s[:10]:
    # for uid in [74153376, 1622441]:
        uid = int(uid)
        fname = join(DATASETS_FOLDER, 'es_twlda100ds_%d.npz' % uid)
        if exists(fname):
            print "%s already exists" % fname
            continue

        print "Processing %d ( f1 %.2f %%)" % (uid, 100 * r)
        # print "Processing %d " % uid

        s = open_session()
        X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(uid)
        # X_train_inds, X_valid_inds, X_test_inds = load_ds_inds(uid)
        
        # X_train_inds = X_train_inds[:10]
        # X_valid_inds = X_valid_inds[:10]
        # X_test_inds = X_test_inds[:10]

        X_train_inds = X_train.index
        X_valid_inds = X_valid.index
        X_test_inds = X_test.index

        print "Loading %d train, %d validation and %d test tweets" % (len(X_train_inds), len(X_valid_inds), len(X_test_inds))
        train_tweets = [s.query(Tweet).get(twid) for twid in X_train_inds]
        valid_tweets = [s.query(Tweet).get(twid) for twid in X_valid_inds]
        test_tweets = [s.query(Tweet).get(twid) for twid in X_test_inds]

        manager = Manager()
        rows_train = manager.list([None] * len(train_tweets))
        rows_valid = manager.list([None] * len(valid_tweets))
        rows_test = manager.list([None] * len(test_tweets))
        lock = manager.Lock()
        counter = manager.Value("i", 0)
        n = len(rows_train) + len(rows_valid) + len(rows_test)

        print "Extracting features"
        NWORKERS = 6
        p = Pool(NWORKERS)
        start = time()

        enumtweets = list(enumerate(train_tweets))
        batch_size = int(ceil(len(train_tweets) * 1.0 / NWORKERS))
        for i in range(NWORKERS):
            its = enumtweets[i * batch_size: (i+1) * batch_size]
            p.apply_async(worker, (its, rows_train, lock, counter, n))

        enumtweets = list(enumerate(valid_tweets))
        batch_size = int(ceil(len(valid_tweets) * 1.0 / NWORKERS))
        for i in range(NWORKERS):
            its = enumtweets[i * batch_size: (i+1) * batch_size]
            p.apply_async(worker, (its, rows_valid, lock, counter, n))

        enumtweets = list(enumerate(test_tweets))
        batch_size = int(ceil(len(test_tweets) * 1.0 / NWORKERS))
        for i in range(NWORKERS):
            its = enumtweets[i * batch_size: (i+1) * batch_size]
            p.apply_async(worker, (its, rows_test, lock, counter, n))

        p.close()
        p.join()

        t = time() - start
        print "Took %.2f secs to process %d tweets" % (t, n)
        print "%.3f sec per tweet" % (t/n)

        def rows_to_csc(rows):
            data = []
            row_ind = []
            col_ind = []
            for i, r in enumerate(rows):
                for j, d in r:
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(d)
            return csc_matrix((data, (row_ind, col_ind)))

        X_train_lda = rows_to_csc(list(rows_train))
        X_valid_lda = rows_to_csc(list(rows_valid))
        X_test_lda = rows_to_csc(list(rows_test))

        np.savez(open(fname,'wb'), X_train_lda, X_valid_lda, X_test_lda)
