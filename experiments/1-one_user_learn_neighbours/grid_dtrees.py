"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.grid_search.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from datasets import load_or_create_dataset
print(__doc__)
import numpy as np


def model_select_dtree(dataset):
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = dataset

    # Set the parameters by cross-validation
    params = {'max_depth':[2,5,10,20],
                'min_samples_split':[2,8,32,128],
                'min_samples_leaf':[1,2,5,10],
                # 'compute_importances':[True],
                # 'max_features': [25, 50, 75, 100, 150]
                # 'max_features': [5, 10, 15]
            }

    scores = [
        'precision',
        'recall',
        'f1'
    ]


    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            DecisionTreeClassifier(),  
            param_grid=params,  # parameters to tune via cross validation
            refit=True,  # fit using all data, on the best detected classifier
            n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
            scoring=score,  # what score are we optimizing?
            cv=StratifiedKFold(y_train, n_folds=3),  # what type of cross validation to use
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
    dataset = load_or_create_dataset()
    model_select_dtree(dataset)
    # model_select_rdf()