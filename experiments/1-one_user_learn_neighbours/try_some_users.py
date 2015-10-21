#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.datasets import load_or_create_dataset, TEST_USERS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import sys

def train_and_evaluate(user_id, username, tweet_count):
    print("==================================")
    print("Loading dataset for user %s (id %d)" % (username, user_id))
    X_train, X_test, y_train, y_test = load_or_create_dataset(user_id)
    ds_size = X_train.shape[0] + X_test.shape[0]
    ds_dimension = X_train.shape[1]
    print("Dataset loaded.")
    print("Size (#tweets): %d" % ds_size)
    print("Dimension (#neighbours): %d" % ds_dimension)

    # weights for class balancing
    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weights = np.array([w0 if x==0 else w1 for x in y_train])

    print("Training RandomForestClassifier")        
    clf = RandomForestClassifier()
    clf_fit = clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))



if __name__ == '__main__':
    # Usage python try_some_users <job_number>
    # jobn = int(sys.argv[1])

    # test_users = TEST_USERS[2 * jobn: 2 * jobn + 2]
    test_users = [x for x in TEST_USERS if x[0] == pendingids[0]]
    for user_id, username, tweet_count in test_users:
        train_and_evaluate(user_id, username, tweet_count)
