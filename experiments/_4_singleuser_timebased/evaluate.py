#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function

from experiments.datasets import load_or_create_dataframe, TEST_USERS
from experiments.utils import *

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tw_dataset.dbmodels import *    
import pickle, os
import numpy as np
import pandas as pd
import sys

def train_on_past_test_with_future(uid, username=None, test_size=0.3, nbuckets=None):
    s = open_session()
    if not username:
        user = s.query(User).get(uid)
        username = user.username
    # Load dataframe and output vector
    print("Loading dataset for user %s (id %d)" % (username, uid))
    Xdf, y = load_or_create_dataframe(uid)

    # Sort tweets by time
    twids = Xdf.index
    ngids = Xdf.columns

    try:
        # Note: this doesn't work when there are too many tweets
        sorted_twids = s.query(Tweet.id).\
                    filter(Tweet.id.in_(twids)).\
                    order_by(Tweet.created_at).all()
        sorted_twids = [x[0] for x in sorted_twids]
    except Exception:
        twtimestamps = {twid: s.query(Tweet).get(twid).created_at for twid in twids}
        sorted_twids = sorted(twids, key=lambda twid: twtimestamps[twid])

    # Split according to proportion
    limit = int(len(sorted_twids) * (1 - test_size))
    train_twids = sorted_twids[:limit]
    test_twids = sorted_twids[limit:]

    X_train = Xdf.loc[train_twids].as_matrix()
    X_test = Xdf.loc[test_twids].as_matrix()
    ydf = pd.DataFrame(data=y, index=twids)
    y_train = ydf.loc[train_twids].as_matrix().transpose()[0]
    y_test = ydf.loc[test_twids].as_matrix().transpose()[0]

    # Train model on past
    # weights for class balancing
    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weights = np.array([w0 if x==0 else w1 for x in y_train])
    if nbuckets:
        print("Transforming features into bucketed format")
        X_train = transform_ngfeats_to_bucketfeats(uid, ngids, X_train, nbuckets)
        X_test = transform_ngfeats_to_bucketfeats(uid, ngids, X_test, nbuckets)

    print("Training RandomForestClassifier on past data")  
    clf = RandomForestClassifier()
    clf_fit = clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_true, y_pred = y_train, clf.predict(X_train)

    print("Scores on training set (past).\n")
    print(classification_report(y_true, y_pred))

    # Evaluate on future
    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set (future).\n")
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    jobn = int(sys.argv[1])
    test_users = TEST_USERS[2 * jobn: 2 * jobn + 2]
    for uid, username, _ in test_users:
        # train_on_past_test_with_future(uid, username)
        train_on_past_test_with_future(uid, username, nbuckets=20)