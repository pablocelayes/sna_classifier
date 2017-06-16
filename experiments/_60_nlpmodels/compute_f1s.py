#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Process, Manager, Pool
from experiments.utils import *
from experiments.datasets import *
from sklearn.metrics import f1_score
import json

import numpy as np
from experiments._60_nlpmodels.fit_social_nlp_models import *
import sys

def scale(X_train, X_test):
    train_size = X_train.shape[0]
    X = np.concatenate((X_train, X_test))
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]

    return X_train, X_test, scaler

def worker(uid, nlp_feats, nlp_model, n_topics, f1s_n, lock):
    """worker function"""
    print "Largamos para %d, %d topics" % (uid, n_topics)

    uid = int(uid)
    clf = load_model(uid, 'svc' + nlp_model, n_topics=n_topics, subfolder='social_nlp')

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataframe(uid)

    X_train_nlp = nlp_feats.loc[X_train.index]
    X_valid_nlp = nlp_feats.loc[X_valid.index]
    X_test_nlp = nlp_feats.loc[X_test.index]

    X_train_combined = np.hstack((X_train, X_train_nlp))
    X_valid_combined = np.hstack((X_valid, X_valid_nlp))
    X_test_combined = np.hstack((X_test, X_test_nlp))

    X_train_combined, X_valid_combined, scaler = scale(X_train_combined, X_valid_combined)
    # Aplicar mismo scaling a X_test
    X_test_combined = scaler.transform(X_test_combined)
    
    y_true, y_pred = y_train, clf.predict(X_train_combined)
    s = f1_score(y_true, y_pred)
    lock.acquire()
    f1s_n['train'].append(s)
    lock.release()
    
    y_true, y_pred = y_valid, clf.predict(X_valid_combined)
    s = f1_score(y_true, y_pred)
    lock.acquire()
    f1s_n['valid'].append(s)
    lock.release()
        
    y_true, y_pred = y_test, clf.predict(X_test_combined)
    s = f1_score(y_true, y_pred)
    lock.acquire()
    f1s_n['test'].append(s)
    lock.release()
    

if __name__ == '__main__':
    nlp_model = sys.argv[1]

    f1s_social = load_nlp_selected_users()

    f1s_combined = {}

    for n_topics in [10, 16, 20]:
        nlp_feats = pd.read_pickle('alltweets_es_%s%d.pickle' % (nlp_model, n_topics))

        uids = [u for u, f1 in f1s_social]

        pool = Pool(processes=7)

        manager = Manager()
        f1s_n = manager.dict()
        lock = manager.Lock()

        for s in 'train valid test'.split():
            f1s_n[s] = []

        for uid in uids:
            pool.apply_async(worker, (uid, nlp_feats, nlp_model, n_topics, f1s_n, lock))
        pool.close()
        pool.join()

        f1s_combined[n_topics] = dict(f1s_n)

    with open('f1s_combined_%s.json' % nlp_model, 'w') as f:
        json.dump(dict(f1s_combined), f)
