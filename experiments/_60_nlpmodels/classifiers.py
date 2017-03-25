# from experiments._1_one_user_learn_neighbours.try_some_users import *
from __future__ import print_function
from tw_dataset.settings import DATASETS_FOLDER
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from os.path import join
import numpy as np
from random import sample
from create_clesa_datasets import *

mpl = log_to_stderr()
mpl.setLevel(logging.ERROR)

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))

def sub_sample_negs_arr(X, y):
    npos = int(sum(y))
    neg_inds = [i for i in range(len(y)) if y[i] == 0]
    pos_inds = [i for i in range(len(y)) if y[i]]
    sample_neg_inds = sample(neg_inds, npos)
    inds = pos_inds + sample_neg_inds

    Xs = X[inds,:]
    ys = y[inds]

    return Xs, ys

def model_select_rdf(dataset, cv=3, n_jobs=6):
    X_train, X_test, y_train, y_test = dataset

    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weight = np.array([w0 if x==0 else w1 for x in y_train])

    # Set the parameters by cross-validation
    params = dict(
        max_depth=[5, 15, None],
        n_estimators=[10, 30, 100],
        class_weight=['balanced_subsample', 'balanced'],
        # sample_weight=[sample_weight]
        max_features=[50, 300, None, 'auto'],
        min_samples_leaf=[1, 3]
    )

    scores = [
        # 'recall',
        'f1',
        # 'precision',
    ]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            RandomForestClassifier(),  
            param_grid=params,  # parameters to tune via cross validation
            refit=True,  # fit using all data, on the best detected classifier
            n_jobs=n_jobs,  # number of cores to use for parallelization; -1 for "all cores"
            scoring=score,  # what score are we optimizing?
            cv=cv,  # what type of cross validation to use
        )

        clf.fit(X_train, y_train)

        print("Best parameters set found on training set:")
        print()
        print(clf.best_params_)

        print("Detailed classification report:")
        print()
        print("Scores on training set.")
        y_true, y_pred = y_train, clf.predict(X_train)
        print(classification_report(y_true, y_pred))
        print()


        print("Scores on test set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    # return clf

def model_select_svc(dataset, cv=3, n_jobs=6):
    X_train, X_test, y_train, y_test = dataset

    # Set the parameters by cross-validation
    parameters = [
        {
         'kernel': ['rbf', 'poly'],
         'gamma': [10, 100, 150, 200],
         'C': [0.01, 0.05, 0.1, 1]
        }
    ]

    scores = [
        # 'precision',
        # 'recall',
        'f1'
    ]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(),  
            param_grid=parameters,  # parameters to tune via cross validation
            refit=True,  # fit using all data, on the best detected classifier
            n_jobs=n_jobs,  # number of cores to use for parallelization; -1 for "all cores"
            scoring=score,  # what score are we optimizing?
            cv=cv,  # what type of cross validation to use
        )

        clf.fit(X_train, y_train)

        print("Best parameters set found on training set:")
        print()
        print(clf.best_params_)

        print("Detailed classification report:")
        print()
        print("Scores on training set.")
        y_true, y_pred = y_train, clf.predict(X_train)
        print(classification_report(y_true, y_pred))
        print()


        print("Scores on test set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

def model_select_sgd(dataset, cv=3, n_jobs=6):
    X_train, X_test, y_train, y_test = dataset

    # Set the parameters by cross-validation
    parameters = [
        {
            'alpha': (0.01, 0.001, 0.00001),
            'penalty': ('l1', 'l2', 'elasticnet'),
            'loss': ('hinge', 'log'),
            'n_iter': (10, 50, 80),
        }
    ]

    scores = [
        # 'precision',
        # 'recall',
        'f1'
    ]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SGDClassifier(),  
            param_grid=parameters,  # parameters to tune via cross validation
            refit=True,  # fit using all data, on the best detected classifier
            n_jobs=n_jobs,  # number of cores to use for parallelization; -1 for "all cores"
            scoring=score,  # what score are we optimizing?
            cv=cv,  # what type of cross validation to use
        )

        clf.fit(X_train, y_train)

        print("Best parameters set found on training set:")
        print()
        print(clf.best_params_)

        print("Detailed classification report:")
        print()
        print("Scores on training set.")
        y_true, y_pred = y_train, clf.predict(X_train)
        print(classification_report(y_true, y_pred))
        print()


        print("Scores on test set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

if __name__ == '__main__':
    # from create_clesa_datasets import *
    # uid=37226353
    uid = 42976687
    dataset = load_clesa_dataset(uid)
    clf = model_select_rdf(dataset)
