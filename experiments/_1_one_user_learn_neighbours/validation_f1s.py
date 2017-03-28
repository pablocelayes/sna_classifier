#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Process, Manager, Pool
from experiments._1_one_user_learn_neighbours.try_some_users import *
from experiments.utils import *
from experiments.datasets import *
from sklearn.metrics import f1_score
import json


def worker(uid, f1s_valid, f1s_testv, lock):
    """worker function"""
    print "Largamos para %d" % uid
    
    try:
        clf = load_model_small(uid, 'svc')
        X_train, X_valid, X_testv, y_train, y_valid, y_testv = load_small_validation_dataframe(uid)
    except Exception as e:
        return

    y_true, y_pred = y_valid, clf.predict(X_valid)
    rs = f1_score(y_true, y_pred)
    lock.acquire()
    f1s_valid[uid] = rs
    lock.release()

    y_true, y_pred = y_testv, clf.predict(X_testv)
    rs = f1_score(y_true, y_pred)
    lock.acquire()
    f1s_testv[uid] = rs
    lock.release()


if __name__ == '__main__':

    uids = [u for u, _, _ in TEST_USERS_ALL]

    pool = Pool(processes=3)

    manager = Manager()
    f1s_valid = manager.dict()
    f1s_testv = manager.dict()
    lock = manager.Lock()

    for uid in uids:
        pool.apply_async(worker, (uid, f1s_valid, f1s_testv, lock))
    pool.close()
    pool.join()

    with open('f1s_valid_svc.json', 'w') as f:
        json.dump(dict(f1s_valid), f)

    with open('f1s_testv_svc.json', 'w') as f:
        json.dump(dict(f1s_testv), f)
