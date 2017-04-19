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
from sklearn.preprocessing import StandardScaler
from __classifiers import model_select_svc
from sklearn.externals import joblib

mpl = log_to_stderr()
mpl.setLevel(logging.ERROR)

PREFIX = '/home/pablo/Proyectos/cogfor/repo/data/es'

tu_path = "/home/pablo/Proyectos/tesiscomp/tw_dataset/active_with_neighbours.json"

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

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))

def scale(X_train, X_test):
    train_size = X_train.shape[0]
    X = np.concatenate((X_train, X_test))
    X = StandardScaler().fit_transform(X)
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]

    return X_train, X_test

def normalize(X_train, X_test):
    train_size = X_train.shape[0]
    X = vstack((X_train, X_test))
    X = _normalize(X)

    X_train = X[:train_size,:]
    X_test = X[train_size:,:]

    return X_train, X_test

def load_dataframe(uid):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ys_%d.pickle" % uid)

    X_train = pd.read_pickle(Xtrain_fname)
    X_valid = pd.read_pickle(Xvalid_fname)
    X_test = pd.read_pickle(Xtest_fname)        
    y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def save_model(clf, user_id, feat_space='', n_topics=None):
    n_topics_str = 't%d' % n_topics if n_topics else ''
    model_path = join(MODELS_FOLDER, "svc_%d%s%s.pickle" % (user_id, feat_space, n_topics_str))
    joblib.dump(clf, model_path)

def load_model(user_id, feat_space='', n_topics=None):
    n_topics_str = 't%d' % n_topics if n_topics else ''
    model_path = join(MODELS_FOLDER, "svc_%d%s%s.pickle" % (user_id, feat_space, n_topics_str))

    clf = joblib.load(model_path)
    return clf

if __name__ == '__main__':
    f1s = load_nlp_selected_users()

    lda_feats = pd.read_pickle('alltweets_es_lda30.pickle')

    # Train classifiers
    for uid, f1 in f1s:
        uid = int(uid)

        print "==============================" 
        print "Processing %d ( f1 %.2f %%)" % (uid, 100 * f1)

        X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataframe(uid)
        
        X_train_lda = lda_feats.loc[X_train.index]
        X_valid_lda = lda_feats.loc[X_valid.index]
        X_test_lda = lda_feats.loc[X_test.index]


        X_train_combined = np.hstack((X_train, X_train_lda))
        X_valid_combined = np.hstack((X_valid, X_valid_lda))
        X_test_combined = np.hstack((X_test, X_test_lda))

        X_train_combined, X_valid_combined = scale(X_train_combined, X_valid_combined)
        ds_comb = (X_train_combined, X_valid_combined, y_train, y_valid)
        comb_clf = model_select_svc(ds_comb, n_jobs=6)
        save_model(comb_clf, uid, 'svclda', n_topics=30)

        print "Results on social model"
        sna_clf = load_model(uid, 'svc')
        evaluate_model(sna_clf, X_train, X_valid, y_train, y_valid)