#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Process, Manager, Pool
from experiments._1_one_user_learn_neighbours.try_some_users import *
from experiments.utils import *
from experiments.datasets import *
from sklearn.metrics import recall_score, f1_score
import json


def worker(uid, recalls_valid, recalls_testv, lock):
    """worker function"""
    print "Largamos para %d" % uid
    s = open_session()
    
    user = s.query(User).get(uid)
    clf = load_model(uid)

    X_train, X_valid, X_testv, y_train, y_valid, y_testv = load_validation_dataframe(uid)
    
    y_true, y_pred = y_valid, clf.predict(X_valid)
    rs = recall_score(y_true, y_pred)
    lock.acquire()
    recalls_valid[uid] = rs
    lock.release()

    y_true, y_pred = y_testv, clf.predict(X_testv)
    rs = recall_score(y_true, y_pred)
    lock.acquire()
    recalls_testv[uid] = rs
    lock.release()


if __name__ == '__main__':

    uids = [u for u, _, _ in TEST_USERS_ALL]

    pool = Pool(processes=3)

    manager = Manager()
    recalls_valid = manager.dict()
    recalls_testv = manager.dict()
    lock = manager.Lock()

    for uid in uids:
        pool.apply_async(worker, (uid, recalls_valid, recalls_testv, lock))
    pool.close()
    pool.join()

    with open('recalls_valid.json', 'w') as f:
        json.dump(dict(recalls_valid), f)

    with open('recalls_testv.json', 'w') as f:
        json.dump(dict(recalls_testv), f)
