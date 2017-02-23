#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.datasets import *
from experiments.utils import *

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tw_dataset.dbmodels import *    
import pickle, os


if __name__ == '__main__':
    # Train on one user
    # uid, username, tweet_count = TEST_USERS[0]
    
    # the one with largest training set
    uid, username, tweet_count = [
        126527644, 
        "Décimo Doctor", 
        714
    ]
    print("Loading dataset for user %s (id %d)" % (username, uid))
    X_train, X_test, y_train, y_test = load_or_create_dataset(uid)

    s = open_session()
    user = s.query(User).get(uid)
    neighbours = get_level2_neighbours(user, s)
    ngids = [str(ng.id) for ng in neighbours]

    X_train = transform_ngfeats_to_bucketfeats(uid, ngids, X_train)
    X_test = transform_ngfeats_to_bucketfeats(uid, ngids, X_test)

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

    print("Detailed classification report:\n")

    y_true, y_pred = y_train, clf.predict(X_train)
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))

    # Test on others
    print("Testing on others")
    other_users = [x for x in TEST_USERS if x[0] != uid]
    for uid, username, tweet_count in other_users:
        print("==================================")
        print("Loading dataset for user %s (id %d)" % (username, uid))
        X_train, X_test, y_train, y_test = load_or_create_dataset(uid)

        user = s.query(User).get(uid)
        neighbours = get_level2_neighbours(user, s)
        ngids = [str(ng.id) for ng in neighbours]

        X_train = transform_ngfeats_to_bucketfeats(uid, ngids, X_train)
        X_test = transform_ngfeats_to_bucketfeats(uid, ngids, X_test)

        ds_size = X_train.shape[0] + X_test.shape[0]
        ds_dimension = X_train.shape[1]
        print("Dataset loaded.")
        print("Size (#tweets): %d" % ds_size)
        print("Dimension (#neighbour buckets): %d" % ds_dimension)

        print("Detailed classification report:\n")

        y_true, y_pred = y_train, clf.predict(X_train)
        print("Scores on training set.\n")
        print(classification_report(y_true, y_pred))

        y_true, y_pred = y_test, clf.predict(X_test)
        print("Scores on test set.\n")
        print(classification_report(y_true, y_pred))
