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

# Low recall users (for neighbor buckets classifier)
# ( <= 80%)
TEST_USERS = [   
    [
        228252737, 
        "LAWRENCE JPD ARABIA ", 
        2523
    ], 
    [
        142800528, 
        "@elprofesionalll", 
        2139
    ], 
    [
        37226353, 
        "Leandro Deyuanini", 
        1871
    ], 

    [
        114582574, 
        "Unión Cívica Radical", 
        689
    ], 
    [
        76684633, 
        "Mario Montoya", 
        129
    ],  
    [
        54987976, 
        "pablorgarcia", 
        126
    ] 
]

def evaluate_combined():
    for nbuckets in [1,2,3,4]:
        print("=============================")
        print("Evaluating for nbuckets=%d" % nbuckets)
        print("Loading dataset...")
        X_train, X_test, y_train, y_test = load_or_create_combined_dataset_small(nbuckets)
        print("OK")

        print("Training model...")
        # weights for class balancing
        w1 = sum(y_train)/len(y_train)
        w0 = 1 - w1
        sample_weight = np.array([w0 if x==0 else w1 for x in y_train])        
        
        clf = RandomForestClassifier()
        clf_fit = clf.fit(X_train, y_train, sample_weight=sample_weight)
        print("OK")

        print("Evaluating on test set...")
        y_true, y_pred = y_test, clf.predict(X_test)
        print("Scores on test set.\n")
        print(classification_report(y_true, y_pred))

def evaluate_on_users(test_users):
    for uid, username, tweet_count in test_users:
        for nbuckets in [1, 5, 10, 20, 30]:
            print("==================================")
            print("Loading dataset for user %s (id %d)" % (username, uid))
            X_train, X_test, y_train, y_test = load_or_create_dataset(uid)

            s = open_session()
            user = s.query(User).get(uid)
            neighbours = get_level2_neighbours(user, s)
            ngids = [str(ng.id) for ng in neighbours]

            print("==================================")
            print("Transforming to %d buckets" % nbuckets)
            X_train = transform_ngfeats_to_bucketfeats(uid, ngids, X_train,
                                                    nmostsimilar=0, nbuckets=nbuckets)
            X_test = transform_ngfeats_to_bucketfeats(uid, ngids, X_test,
                                                    nmostsimilar=0, nbuckets=nbuckets)

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
    import sys
    n = int(sys.argv[1])
    evaluate_on_users([TEST_USERS[n]])