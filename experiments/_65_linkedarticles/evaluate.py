#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.svm import SVC

# from gensim import matutils

import pickle, os
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import json

# from extractors import EsaFeatureExtractor
from tw_dataset.settings import PREFIX

# Low recall users (for neighbor buckets classifier)
# ( <= 80%)
USER = [37226353, "Leandro Deyuanini", 1871]
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

# Extract features
def load_esa_dataset(uid, lang='es'):
    fname = 'esads_%d.npz' % uid
    z = np.load(open(fname,'rb'))
    X = z['arr_0']
    y = z['arr_1']

    return X, y


def train_and_evaluate(user_id, username):
    print("==================================")
    print("Loading dataset for user %s (id %d)" % (username, user_id))
    X, y = load_esa_dataset(uid)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    ds_size = X_train.shape[0] + X_test.shape[0]
    ds_dimension = X_train.shape[1]
    print("Dataset loaded.")
    print("Size (#tweets): %d" % ds_size)
    print("Dimension (#ESA features): %d" % ds_dimension)

    # weights for class balancing
    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weights = np.array([w0 if x==0 else w1 for x in y_train])

    print("Training RandomForestClassifier")        
    # clf = RandomForestClassifier()
    # clf = SVC(kernel="linear", C=0.025, class_weight='auto')
    clf = SVC(gamma=2, C=1, class_weight='auto')
    # clf = DecisionTreeClassifier()
    
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))

def job_train_and_evaluate():
    import sys
    n = int(sys.argv[1])
    uid, username, _ = TEST_USERS[n]

    train_and_evaluate(uid)

if __name__ == '__main__':
    for (uid, username, _) in TEST_USERS:
        train_and_evaluate(uid, username)
