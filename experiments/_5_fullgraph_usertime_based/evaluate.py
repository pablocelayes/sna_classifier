#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function

from experiments.datasets import *
from experiments.utils import *
from sklearn.ensemble import RandomForestClassifier


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weight = np.array([w0 if x==0 else w1 for x in y_train]) 

    clf.fit(X_train, y_train, sample_weight=sample_weight)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    y_true, y_pred = y_train, clf.predict(X_train)
    print(classification_report(y_true, y_pred))

    print("Scores on test set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    X_train, y_train = load_large_dataset_full()
    X_test, y_test = load_large_dataset_full(set_type="testtime")

    # clf = RandomForestClassifier()
    clf = SVC(kernel="rbf", gamma=1, C=0.1, class_weight='auto') # <---- ¡para atrás!

    train_and_evaluate(clf, X_train, X_test, y_train, y_test)
