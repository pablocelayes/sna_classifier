#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.relatedness_calculator import finite_katz_measures
from experiments.datasets import *
from experiments.utils import *
from tw_dataset.dbmodels import *
import pickle, os
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

def get_recall_errors(clf, X, y):
    y_true, y_pred = y, clf.predict(X)
    recall_error_mask = np.logical_and(y_true, 1 - y_pred)
    X_recall_errors = X.iloc[recall_error_mask,:]

    return X_recall_errors

def get_zeros(X, y):
    zero_mask = y == 0
    X_zero = X.iloc[zero_mask,:]

    return X_zero

def fit_balanced(clf, X_train, y_train, extra_weight_ones=1.0):
    w1 = sum(y_train)/len(y_train)
    w1 *= extra_weight_ones
    w0 = 1 - w1
    sample_weight = np.array([w0 if x==0 else w1 for x in y_train])

    clf.fit(X_train, y_train, sample_weight=sample_weight)

if __name__ == '__main__':
    uid=37226353

    X_train, X_test, y_train, y_test = load_or_create_dataframe(uid)

    clf = DecisionTreeClassifier()
    fit_balanced(clf, X_train, y_train)
    # yt, yp = y_train, clf.predict(X_train)
    # classification_report(yt, yp)

    X_recall_errors = get_recall_errors(clf, X_train, y_train)
    N = len(X_recall_errors)

    # TODO: hacer sampling más inteligente
    X_zero = get_zeros(X_train, y_train).iloc[:N,:]

    # Training back off model to separate recall errors
    # from zeros
    X_train_re = pd.concat((X_recall_errors, X_zero))
    y_train_re = np.array([1] * len(X_recall_errors) + [0] * len(X_zero))

    # clf_re = RandomForestClassifier()
    # clf_re = SVC(kernel="rbf", gamma=1, C=0.1, class_weight='auto')
    # fit_balanced(clf_re, X_train_re, y_train_re, extra_weight_ones=10)
    
    # Nota: acá RDF, Dtree, SVC, anduvieron todos muy mal
    print("Trying some alternative models to separate recall errors...")
    print("(With balanced classes)")
    clfs = {
        "BernoulliNB": BernoulliNB(),
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "RandomForestClassifier": RandomForestClassifier(),
        "SVC_RBF": SVC(kernel="rbf", gamma=1, C=0.1, class_weight='auto')
    }

    for name, clf_re in clfs.items():
        print("====================")
        print(name)
        clf_re.fit(X_train_re, y_train_re)
        yt, yp = y_train_re, clf_re.predict(X_train_re)
        print(classification_report(yt, yp))

    # TODO: implementar modelo completo con back off y evaluar en test set

