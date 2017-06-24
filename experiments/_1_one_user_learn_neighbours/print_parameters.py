#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Process, Manager, Pool
from experiments._1_one_user_learn_neighbours.fit_social_models import *
from experiments.utils import *
from experiments.datasets import *
from sklearn.metrics import f1_score, precision_score, recall_score
import json


def worker(uid, f1s_train, f1s_valid, f1s_testv, precisions_train, precisions_valid, precisions_testv,
    recalls_train, recalls_valid, recalls_testv, lock):
    """worker function"""
    print "Largamos para %d" % uid
    
    clf = load_model(uid, 'svc')
    X_train, X_valid, X_testv, y_train, y_valid, y_testv = load_dataframe(uid)

    y_true, y_pred = y_train, clf.predict(X_train)
    lock.acquire()
    f1s_train[uid] = f1_score(y_true, y_pred)
    precisions_train[uid] = precision_score(y_true, y_pred)
    recalls_train[uid] = recall_score(y_true, y_pred)
    lock.release()

    y_true, y_pred = y_valid, clf.predict(X_valid)
    lock.acquire()
    f1s_valid[uid] = f1_score(y_true, y_pred)
    precisions_valid[uid] = precision_score(y_true, y_pred)
    recalls_valid[uid] = recall_score(y_true, y_pred) 
    lock.release()

    y_true, y_pred = y_testv, clf.predict(X_testv)
    lock.acquire()
    f1s_testv[uid] = f1_score(y_true, y_pred)
    precisions_testv[uid] = precision_score(y_true, y_pred)
    recalls_testv[uid] = recall_score(y_true, y_pred)
    lock.release()

if __name__ == '__main__':

    uids = [u for u in TEST_USERS_ALL]

    for uid in uids:
        