"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.model_selection.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from hyperopt import hp, fmin, tpe


# from datasets import load_or_create_dataset
from experiments.datasets import load_or_create_combined_dataset_small
from experiments.utils import load_ig_graph

# print(__doc__)
import numpy as np
import xgboost as xgb

import os
import pickle


def model_select_svc(dataset):
    X_train, X_test, y_train, y_test = dataset

    # Set the parameters by cross-validation
    parameters = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'colsample_bytree': [0.3, 0.5, 0.7]
    }

    print(f"# Tuning hyper-parameters for f1 score\n")

    classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    folds = 3
    param_comb = 30
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    clf = RandomizedSearchCV(
        classifier,
        param_distributions=parameters,
        n_iter=param_comb,
        scoring='f1',
        n_jobs=16,
        cv=skf.split(X_train, y_train),
        verbose=7,
        random_state=1001
    )


    # clf.fit(X_train[:100], y_train[:100])
    clf.fit(X_train, y_train)

    print("Best parameters set found on training set:\n")
    print(clf.best_params_)

    print("Detailed classification report:\n")
    print("Scores on training set.")
    y_true, y_pred = y_train, clf.predict(X_train)
    print(classification_report(y_true, y_pred))
    print()

    print("Scores on test set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

params_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
    # A problem with max_depth casted to float instead of int with
    # the hp.quniform method.
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'tree_method': 'exact',
    'nbuckets': hp.quniform('nbuckets', 0, 50, 1),
    'nmostsimilar': hp.quniform('nbuckets', 5, 50, 1),
    'silent': 1,
    'seed': random_state
}

def objective(params):


if __name__ == '__main__':
    # dataset = load_or_create_dataset(37226353)
    # dataset = load_or_create_combined_dataset_small(nmostsimilar=30, nbuckets=20, include_activity_rank=True)
    print("Loading user graph")
    g = load_ig_graph()

    # TODO: save to pickle
    fname = "centralities.pickle"
    if os.path.exists(fname):
        print("Loading pre-computed centralities")
        with open(fname, 'rb') as f:
            centralities = pickle.load(f)
    else:
        print("Computing centralities")
        centralities = g.pagerank(), g.betweenness(), g.closeness(), g.eigenvector_centrality(), g.eccentricity()
        with open(fname, 'wb') as f:
            pickle.dump(centralities, f)

    print("Computing bucketized dataset")
    dataset = load_or_create_combined_dataset_small(g, centralities, nmostsimilar=10, nbuckets=10, include_activity_rank=True)
    # dataset = load_or_create_combined_dataset_small(g, centralities, nmostsimilar=10, nbuckets=10, include_activity_rank=True, n_users=4)

    print("Searching model hparams")
    model_select_svc(dataset)