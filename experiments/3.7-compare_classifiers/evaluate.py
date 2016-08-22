#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.relatedness_calculator import finite_katz_measures
from experiments.datasets import *
from experiments.utils import *

from sklearn import datasets
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

from tw_dataset.dbmodels import *
import pickle, os
import numpy as np

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025, class_weight='auto'),
    SVC(kernel="rbf", gamma=1, C=0.1, class_weight='auto'),
    DecisionTreeClassifier(class_weight='auto'),
    RandomForestClassifier(class_weight='auto'),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

def evaluate_combined_small(njob):
    nbuckets_values = [5]
    for nbuckets in nbuckets_values:
        print("=============================")
        print("Evaluating for nbuckets=%d" % nbuckets)
        print("Loading dataset...")
        X_train, X_test, y_train, y_test = load_or_create_combined_dataset_small(nbuckets)
        print("OK")

        w1 = sum(y_train)/len(y_train)
        w0 = 1 - w1
        sample_weight = np.array([w0 if x==0 else w1 for x in y_train])        
        
        for name, clf in zip(names, classifiers)[2 * njob: 2 * njob + 1]:
            print("--------------------")
            print("Training %s" % name)
            clf.fit(X_train, y_train)
            
            y_true, y_pred = y_test, clf.predict(X_test)
            print("Scores on test set.\n")
            print(classification_report(y_true, y_pred))

def evaluate_individual(uid, njob):
    print("=============================")
    print("Evaluating for uid=%d" % uid)
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_or_create_dataset(uid)
    print("OK")

    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weight = np.array([w0 if x==0 else w1 for x in y_train])        
    
    for name, clf in zip(names, classifiers)[2 * njob: 2 * njob + 2]:
        print("--------------------")
        print("Training %s" % name)
        try:
            clf.fit(X_train, y_train, sample_weight=sample_weight)            
        except Exception:
            print("Doesn't accept sample weight, calling without...")
            clf.fit(X_train, y_train)
        print("OK")
        
        y_true, y_pred = y_test, clf.predict(X_test)
        print("Scores on test set.\n")
        print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    import sys
    njob = int(sys.argv[1])
    evaluate_individual(37226353, njob)
