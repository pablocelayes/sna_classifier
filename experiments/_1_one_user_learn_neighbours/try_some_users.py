#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.datasets import (load_or_create_dataframe, load_small_validation_dataframe,
                                TEST_USERS_ALL)

MODELS_FOLDER = "/Users/pablofreetime/Proyectos/sna_classifier/models"

from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib

import numpy as np
import sys
from os.path import join
from multiprocessing import Pool

from .classifiers import model_select_rdf, model_select_svc

def train_and_evaluate(user_id, clf_class=RandomForestClassifier):
    print("==================================")
    print("Loading dataframe for user id %d" % user_id)
    # X_train, X_test, y_train, y_test = load_or_create_dataframe(user_id)

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(user_id)

    ds_size = X_train.shape[0] + X_test.shape[0]
    ds_dimension = X_train.shape[1]
    print("dataframe loaded.")
    print("Size (#tweets): %d" % ds_size)
    print("Dimension (#neighbours): %d" % ds_dimension)

    # weights for class balancing
    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weights = np.array([w0 if x==0 else w1 for x in y_train])

    print("Training %s" % clf_class.__name__)
    clf = clf_class()     
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    # evaluate_model(clf, X_train, X_test, y_train, y_test)
    evaluate_model(clf, X_train, X_valid, y_train, y_valid)

    return clf

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))



def save_model(clf, user_id):
    model_path = join(MODELS_FOLDER, "rdf_%d.pickle" % user_id)
    joblib.dump(clf, model_path)

def load_model(user_id):
    model_path = join(MODELS_FOLDER, "rdf_%d.pickle" % user_id)
    clf = joblib.load(model_path)
    return clf


def save_model_small(clf, user_id, model_type, feat_space='', n_topics=None):
    n_topics_str = 't%d' % n_topics if n_topics else ''
    model_path = join(MODELS_FOLDER, "%s_%d_small_%s%s.pickle" % (model_type, user_id, feat_space, n_topics_str))
    joblib.dump(clf, model_path)

def load_model_small(user_id, model_type):
    model_path = join(MODELS_FOLDER, "%s_%d_small_.pickle" % (model_type, user_id))

    clf = joblib.load(model_path)
    return clf


def worker(user_id):
    try:
        # clf = train_and_evaluate(user_id)
        X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(user_id)
        dataset = X_train, X_valid, y_train, y_valid
                
        clf = model_select_svc(dataset)
        save_model_small(clf, user_id, 'svc')
    except Exception as e:
        # print(e)
        print(f"An exception occurred for {user_id}")


if __name__ == '__main__':

    # See which users are pending
    pending_user_ids = []
    for user_id, username, _ in TEST_USERS_ALL:
        try:
            load_model_small(user_id, 'svc')
        except IOError:
            print(user_id)
            pending_user_ids.append(user_id)

    print(f"{len(pending_user_ids)} pending models will be trained")

    pool = Pool(processes=6)
    for user_id in pending_user_ids:
        pool.apply_async(worker, (user_id,))
    pool.close()
    pool.join()

    # pending_user_ids = [uid for uid,_,_ in TEST_USERS_ALL]

    # for user_id in pending_user_ids:
    # for user_id in [76684633]:
    #     print(user_id)
    #     worker(user_id)

    # worker(117335842)        



# Modelos elegidos con RandomForestClassifier pelado