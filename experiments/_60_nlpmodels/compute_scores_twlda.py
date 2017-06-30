#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Process, Manager, Pool
from experiments._1_one_user_learn_neighbours.fit_social_models import *
from experiments._60_nlpmodels.fit_social_twlda_models import *

from experiments.utils import *
# from experiments.datasets import *
from sklearn.metrics import f1_score, precision_score, recall_score
import json

MODELS_FOLDER = "/media/pablo/data/Tesis/models/old/"

N_TOPICS = 20

NLP_FEATS = pd.read_pickle(join(TM_MODELS_PATH, "./alltweets_es_twlda%d.pickle" % N_TOPICS))

def worker(uid, f1s_train, f1s_valid, f1s_test,
    precisions_train, precisions_valid, precisions_test,
    recalls_train, recalls_valid, recalls_test, lock):
    """worker function"""
    print "Largamos para %d" % uid
    
    try:
        clf = load_model(uid, 'svctwlda', n_topics=N_TOPICS, subfolder='social_nlp')
    except Exception as e:
        print "Falta un modelo"
        return

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataframe(uid)

    X_train_nlp = NLP_FEATS.loc[X_train.index]
    X_valid_nlp = NLP_FEATS.loc[X_valid.index]
    X_test_nlp = NLP_FEATS.loc[X_test.index]

    # TODO: solve missing tweets
    X_train_nlp[np.isnan(X_train_nlp)] = 0
    X_valid_nlp[np.isnan(X_valid_nlp)] = 0
    X_test_nlp[np.isnan(X_test_nlp)] = 0

    X_train_combined = np.hstack((X_train, X_train_nlp))
    X_valid_combined = np.hstack((X_valid, X_valid_nlp))
    X_test_combined = np.hstack((X_test, X_test_nlp))

    train_size = X_train.shape[0]
    X = np.concatenate((X_train_combined, X_valid_combined))

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X_train_combined = X[:train_size,:]
    # X_valid_combined = X[train_size:,:]
    # X_test_combined = scaler.transform(X_test_combined)

    X_train, X_valid, X_test = X_train_combined, X_valid_combined, X_test_combined

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

    y_true, y_pred = y_test, clf.predict(X_test)
    lock.acquire()
    f1s_test[uid] = f1_score(y_true, y_pred)
    precisions_test[uid] = precision_score(y_true, y_pred)
    recalls_test[uid] = recall_score(y_true, y_pred)
    lock.release()

if __name__ == '__main__':

    f1s_social = load_nlp_selected_users()
    uids = [int(u) for u, f1 in f1s_social]

    pool = Pool(processes=7)

    manager = Manager()

    f1s_train = manager.dict()
    f1s_valid = manager.dict()
    f1s_test = manager.dict()

    precisions_train = manager.dict()
    precisions_valid = manager.dict()
    precisions_test = manager.dict()

    recalls_train = manager.dict()
    recalls_valid = manager.dict()
    recalls_test = manager.dict()

    lock = manager.Lock()

    for uid in uids:
        pool.apply_async(worker, (uid, f1s_train, f1s_valid, f1s_test,
                                       precisions_train, precisions_valid, precisions_test,
                                       recalls_train, recalls_valid, recalls_test,
                                       lock))
    pool.close()
    pool.join()

    with open('scores/tw_t%d_f1s_train_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(f1s_train), f)

    with open('scores/tw_t%d_f1s_valid_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(f1s_valid), f)

    with open('scores/tw_t%d_f1s_test_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(f1s_test), f)

    with open('scores/tw_t%d_precisions_train_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(precisions_train), f)

    with open('scores/tw_t%d_precisions_valid_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(precisions_valid), f)

    with open('scores/tw_t%d_precisions_test_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(precisions_test), f)

    with open('scores/tw_t%d_recalls_train_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(recalls_train), f)

    with open('scores/tw_t%d_recalls_valid_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(recalls_valid), f)

    with open('scores/tw_t%d_recalls_test_svc.json' % N_TOPICS, 'w') as f:
        json.dump(dict(recalls_test), f)
