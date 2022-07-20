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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.utils import check_random_state

from hyperopt import hp, fmin, tpe, Trials


# from datasets import load_or_create_dataset
from experiments.datasets import load_or_create_combined_dataset_small
from experiments.utils import load_ig_graph

# print(__doc__)
import numpy as np
import xgboost as xgb

import os
import pickle
from datetime import datetime


print("Loading user graph")
g = load_ig_graph()


params_space = {
    'n_estimators': hp.choice('n_estimators', np.arange(50, 1000, 1, dtype=int)),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'nbuckets': hp.choice('nbuckets', np.arange(0, 51, 5, dtype=int)),
    'nmostsimilar': hp.choice('nmostsimilar', np.arange(5, 51, 5, dtype=int)),
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'tree_method': 'exact',
    'nthread': -1,
    'silent': 1,
    'seed': 42,
    'g': g
}

TESTING = False
# TESTING = True

if TESTING:
    N_TRIALS = 2
    N_USERS = 4
    params_space['n_estimators'] = hp.choice('n_estimators', np.arange(2, 10, 1, dtype=int))
else:
    # N_TRIALS = 20
    N_TRIALS = 50
    N_USERS=None


def objective(params):
    g, nmostsimilar, nbuckets = params["g"], params["nmostsimilar"], params["nbuckets"]
    del params["g"]
    del params["nmostsimilar"]
    del params["nbuckets"]

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
    dataset = load_or_create_combined_dataset_small(g, centralities,
                                                    n_users=N_USERS,
                                                    nmostsimilar=nmostsimilar,
                                                    nbuckets=nbuckets,
                                                    include_activity_rank=True)

    X_train, X_test, y_train, y_test = dataset

    # Set the parameters by cross-validation
    clf = xgb.XGBClassifier(use_label_encoder=False, **params)

    # Handle randomization by shuffling when creating folds. In reality, we probably want a better
    # strategy for managing randomization than the fixed `RandomState` instance generated above.
    # cv = KFold(n_splits=4, random_state=2022, shuffle=True)
    cv = StratifiedKFold(n_splits=4, random_state=2022, shuffle=True)

    # Calculate f1 score for each fold. Since `n_splits` is 4, `f1_trial` will be an
    # array of size 4 with each element representing the f1 score for a fold.
    f1_trial_cv = cross_val_score(
        clf,
        X_train,
        y_train,
        scoring=make_scorer(f1_score),
        cv=cv,
    )

    return -np.mean(f1_trial_cv)

if __name__ == '__main__':
    # `hyperopt` tracks the results of each iteration in this `Trials` object. We’ll be collecting the
    # data that we will use for visualization from this object.
    trials = Trials()

    # reproducibility!
    rstate = np.random.default_rng(2022)
    best = fmin(objective, params_space, algo=tpe.suggest, max_evals=N_TRIALS, trials=trials, rstate=rstate)

    print(best)

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M")
    with open(f"trials_{ts}.pickle", "wb") as f:
        pickle.dump(trials, f)

    # Refit with best parameters, report results on test set
    # clf.fit(X_train, y_train)
    #
    # print("Scores on test set.\n")
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
