#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.datasets import (load_or_create_dataframe,
    TEST_USERS, TEST_USERS_2, TEST_USERS_3, TEST_USERS_4, TEST_USERS_ALL    
)

# TEST_USERS = TEST_USERS + TEST_USERS_2 + TEST_USERS_3 + TEST_USERS_4

TEST_USERS = TEST_USERS_ALL

MODELS_FOLDER = "/media/pablo/data/Tesis/models/"

from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import numpy as np
import sys
from os.path import join
from multiprocessing import Pool


def train_and_evaluate(user_id, clf_class=RandomForestClassifier):
    print("==================================")
    print("Loading dataframe for user id %d" % user_id)
    X_train, X_test, y_train, y_test = load_or_create_dataframe(user_id)
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

    evaluate_model(clf, X_train, X_test, y_train, y_test)

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

def worker(user_id):
    try:
        clf = train_and_evaluate(user_id)
        save_model(clf, user_id)
    except Exception as e:
        print(e)


if __name__ == '__main__':

    # See which users are pending
    pending_user_ids = []
    for user_id, username, _ in TEST_USERS_ALL:
        try:
            load_model(user_id)
        except IOError:
            pending_user_ids.append(user_id)


    # pool = Pool(processes=2)
    # for user_id in pending_user_ids:
    #     pool.apply_async(worker, (user_id,))
    # pool.close()
    # pool.join()

    for user_id in pending_user_ids:
        worker(user_id)
