#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from experiments.datasets import *
from sklearn.metrics import classification_report
import random


# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.lda import LDA
from sklearn.qda import QDA

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025, class_weight='auto'),
    SVC(kernel="rbf", gamma=1, C=0.1, class_weight='auto'),
    DecisionTreeClassifier(class_weight='auto'),
    RandomForestClassifier(class_weight='auto'),
    BernoulliNB(),
    LDA(),
    QDA()]

def get_zeros(X, y):
    zero_mask = y == 0
    X_zero = X.iloc[zero_mask,:]

    return X_zero


class BackoffClassifier(object):
    """docstring for BackoffClassifier"""
    def __init__(self, classifier=RandomForestClassifier()):
        self.clf_main = DecisionTreeClassifier()
        self.clf_backoff = classifier

    def fit(self, X_train, y_train):
        # Train balanced main model
        w1 = sum(y_train)/len(y_train)
        w0 = 1 - w1
        sample_weight = np.array([w0 if x==0 else w1 for x in y_train])

        self.clf_main.fit(X_train, y_train, sample_weight=sample_weight)

        # Train resampled backoff model to separate recall errors
        # from true zeros
        X_recall_errors = self.get_recall_errors(X_train, y_train)
        N = len(X_recall_errors)

        # TODO: hacer sampling más inteligente
        X_zero = get_zeros(X_train, y_train).iloc[:3 * N,:]

        X_train_re = pd.concat((X_recall_errors, X_zero))
        y_train_re = np.array([1] * len(X_recall_errors) + [0] * len(X_zero))
        
        w1 = sum(y_train_re)/len(y_train_re)
        extra_weight_ones = 2.0
        w1 *= extra_weight_ones

        w0 = 1 - w1
        sample_weight = np.array([w0 if x==0 else w1 for x in y_train_re])

        try:
            self.clf_backoff.fit(X_train_re, y_train_re, sample_weight=sample_weight)
        except Exception:
            print("Doesn't accept sample weight, calling without...")
            self.clf_backoff.fit(X_train_re, y_train_re)
        print("OK")


    def get_recall_errors(self, X, y):
        y_true, y_pred = y, self.clf_main.predict(X)
        recall_error_mask = np.logical_and(y_true, 1 - y_pred)
        X_recall_errors = X.iloc[recall_error_mask,:]

        return X_recall_errors

    def predict(self, X):
        nrows = X.shape[0]
        preds = np.empty(nrows)
        for i in range(nrows):
            x = X.iloc[i,:]
            main_pred = self.clf_main.predict(x)

            if main_pred:
                preds[i] = main_pred
            else:
                random.randint(0,1)
                backoff_pred = self.clf_backoff.predict(x)
                preds[i] = backoff_pred

        return preds


def train_and_evaluate(user_id, username, clf):
    print("==================================")
    print("Loading dataset for user %s (id %d)" % (username, user_id))
    X_train, X_test, y_train, y_test = load_or_create_dataframe(user_id)

    clf.fit(X_train, y_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    y_true, y_pred = y_train, clf.predict(X_train)
    print(classification_report(y_true, y_pred))

    print("Scores on test set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

def try_on_all_users():
    for user_id, username, _ in TEST_USERS:
        clf = BackoffClassifier()
        train_and_evaluate(user_id, username)

def compare_classifiers():
    import sys
    njob = int(sys.argv[1])
    user_id = 37226353
    username = "Leandro Deyuanini"
    for clf in classifiers[njob: njob + 1]:
        class_name = str(clf.__class__)
        clf_name = class_name.split('.')[-1]
        print("Trying backoff classifier %s" % clf_name)
        backoff_clf = BackoffClassifier(clf)
        train_and_evaluate(user_id, username, backoff_clf)

if __name__ == '__main__':
    compare_classifiers()
