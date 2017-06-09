#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Process, Manager, Pool
from experiments._1_one_user_learn_neighbours.fit_social_models import *
from experiments.utils import *
from experiments.datasets import *
from sklearn.metrics import f1_score, precision_score, recall_score
import json

MODELS_FOLDER = "/media/pablo/data/Tesis/models/old/"

def load_model_small(uid):
    model_path = join(MODELS_FOLDER, "svc_%d_small.pickle" % uid)
    clf = joblib.load(model_path)

    return clf

def worker(uid, f1s_train, f1s_valid, f1s_testv, precisions_train, precisions_valid, precisions_testv,
    recalls_train, recalls_valid, recalls_testv, lock):
    """worker function"""
    print "Largamos para %d" % uid
    
    clf = load_model_small(uid)
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

    uids = [u[0] for u in TEST_USERS_ALL]

    pool = Pool(processes=7)

    manager = Manager()

    f1s_train = manager.dict()
    f1s_valid = manager.dict()
    f1s_testv = manager.dict()

    precisions_train = manager.dict()
    precisions_valid = manager.dict()
    precisions_testv = manager.dict()

    recalls_train = manager.dict()
    recalls_valid = manager.dict()
    recalls_testv = manager.dict()

    lock = manager.Lock()

    for uid in uids:
        pool.apply_async(worker, (uid, f1s_train, f1s_valid, f1s_testv,
                                       precisions_train, precisions_valid, precisions_testv,
                                       recalls_train, recalls_valid, recalls_testv,
                                       lock))
    pool.close()
    pool.join()

    with open('scores/f1s_train_svc.json', 'w') as f:
        json.dump(dict(f1s_valid), f)

    with open('scores/f1s_valid_svc.json', 'w') as f:
        json.dump(dict(f1s_valid), f)

    with open('scores/f1s_testv_svc.json', 'w') as f:
        json.dump(dict(f1s_testv), f)

    with open('scores/precisions_train_svc.json', 'w') as f:
        json.dump(dict(precisions_valid), f)

    with open('scores/precisions_valid_svc.json', 'w') as f:
        json.dump(dict(precisions_valid), f)

    with open('scores/precisions_testv_svc.json', 'w') as f:
        json.dump(dict(precisions_testv), f)

    with open('scores/recalls_train_svc.json', 'w') as f:
        json.dump(dict(recalls_valid), f)

    with open('scores/recalls_valid_svc.json', 'w') as f:
        json.dump(dict(recalls_valid), f)

    with open('scores/recalls_testv_svc.json', 'w') as f:
        json.dump(dict(recalls_testv), f)
