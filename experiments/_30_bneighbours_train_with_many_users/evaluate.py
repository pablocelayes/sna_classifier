#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.relatedness_calculator import finite_katz_measures
from experiments.datasets import *
from experiments.utils import *

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier

from tw_dataset.dbmodels import *    
import pickle, os
import numpy as np

# TEST_USERS = TEST_USERS[:3]
NBUCKETS = 20

if __name__ == '__main__':
    print("###################################")
    print("Transforming and combining training sets")
    
    s = open_session()
    X_train = None
    y_train = None
    for uid, username, tweet_count in TEST_USERS:
        print("==================================")
        print("Loading training set for user %s (id %d)" % (username, uid))
        u_X_train, _, u_y_train, _ = load_or_create_dataset(uid)

        ds_size, ds_dimension = u_X_train.shape
        print("Size (#tweets): %d" % ds_size)
        print("Dimension (#neighbours): %d" % ds_dimension)
        user = s.query(User).get(uid)
        neighbours = get_level2_neighbours(user, s)
        ngids = [str(ng.id) for ng in neighbours]

        u_X_train = transform_ngfeats_to_bucketfeats(uid, ngids, u_X_train, NBUCKETS)
        if X_train is None:
            X_train = u_X_train
            y_train = u_y_train
        else:
            X_train = np.vstack((X_train, u_X_train))
            y_train = np.hstack((y_train, u_y_train))

    ds_size, ds_dimension = X_train.shape
    print("==================================")    
    print("Combined training set created.")
    print("Size (#tweets): %d" % ds_size)
    print("Dimension (#neighbour buckets): %d" % ds_dimension)


    print("###################################")
    print("Training RandomForestClassifier on combined training set")
    # weights for class balancing
    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weights = np.array([w0 if x==0 else w1 for x in y_train])        
    
    clf = RandomForestClassifier()
    clf_fit = clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_true, y_pred = y_train, clf.predict(X_train)
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    print("###################################")
    print("Evaluating on individual test sets")
    for uid, username, tweet_count in TEST_USERS:
        print("==================================")
        print("Loading dataset for user %s (id %d)" % (username, uid))
        _, X_test, _, y_test = load_or_create_dataset(uid)

        user = s.query(User).get(uid)
        neighbours = get_level2_neighbours(user, s)
        ngids = [str(ng.id) for ng in neighbours]

        X_test = transform_ngfeats_to_bucketfeats(uid, ngids, X_test, NBUCKETS)

        ds_size, ds_dimension = X_test.shape
        print("Transformed test set created.")
        print("Size (#tweets): %d" % ds_size)
        print("Dimension (#neighbour buckets): %d" % ds_dimension)

        y_true, y_pred = y_test, clf.predict(X_test)
        print("Scores on test set.\n")
        print(classification_report(y_true, y_pred))
