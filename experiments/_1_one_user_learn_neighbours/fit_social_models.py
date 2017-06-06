#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.datasets import (load_dataframe, TEST_USERS_ALL)

MODELS_FOLDER = "/media/pablo/data/Tesis/models/"

from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import numpy as np
import sys
from os.path import join
from multiprocessing import Pool

from classifiers import model_select_rdf, model_select_svc


def evaluate_model(clf, X_train, X_test, y_train, y_test):
    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))


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
    # See which users are pending
    pending_user_ids = []
    for user_id in TEST_USERS_ALL:
        try:
            load_model(user_id, 'svc')
        except IOError:
            pending_user_ids.append(user_id)


    for user_id in pending_user_ids:
        print(user_id)
        try:
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataframe(user_id)
            dataset = X_train, X_valid, y_train, y_valid        
            clf = model_select_svc(dataset, n_jobs=4)
            save_model(clf, user_id, 'svc')
        except Exception as e:
            print(e)
