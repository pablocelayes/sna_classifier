#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.datasets import *

MODELS_FOLDER = "/media/pablo/data/Tesis/models/old"

from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import numpy as np
import sys
from os.path import join
from multiprocessing import Pool

from classifiers import model_select_rdf, model_select_svc

from experiments._1_one_user_learn_neighbours.try_some_users import evaluate_model



def save_model(clf, user_id):
    model_path = join(MODELS_FOLDER, "rdf_%d.pickle" % user_id)
    joblib.dump(clf, model_path)

def load_model(user_id):
    model_path = join(MODELS_FOLDER, "rdf_%d.pickle" % user_id)
    clf = joblib.load(model_path)
    return clf


def save_model_small(clf, user_id, model_type, feat_space='', n_topics=None):
    n_topics_str = 't%d' % n_topics if n_topics else ''
    model_path = join(MODELS_FOLDER, "%s_%d_small_%s%s.pickle" % (model_type, user_id, feat_space, n_topics_str))
    joblib.dump(clf, model_path)

def load_model_small(user_id, model_type, feat_space='', n_topics=None):
    n_topics_str = 't%d' % n_topics if n_topics else ''
    model_path = join(MODELS_FOLDER, "%s_%d_small_%s%s.pickle" % (model_type, user_id, feat_space, n_topics_str))

    clf = joblib.load(model_path)
    return clf

def load_Xallpop():
    s = open_session()
    q = s.query(Tweet.id, Tweet.favorite_count, Tweet.retweet_count)
    df = pd.DataFrame(q.all())
    df.index = df.id
    del df['id']

    return df

from sklearn.linear_model import LogisticRegression


def convert_to_popularity(X, Xallpop):
    tweet_ids = X.index
    Xpop = Xallpop.loc[tweet_ids]
    return Xpop



if __name__ == '__main__':

    from tw_dataset.dbmodels import *
    import pandas as pd

    Xallpop = load_Xallpop()
    
    # See which users are pending
    pending_user_ids = []
    for user_id, username, _ in TEST_USERS_ALL:
        try:
            load_model_small(user_id, 'svcpop')
        except IOError:
            pending_user_ids.append(user_id)

    for uid in pending_user_ids:
        print(uid)
        try:
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataframe(uid)

            X_train_pop = convert_to_popularity(X_train, Xallpop)
            X_valid_pop = convert_to_popularity(X_valid, Xallpop)
            X_test_pop = convert_to_popularity(X_test, Xallpop)

            clfpop = LogisticRegression(class_weight='balanced')
            clfpop.fit(X_train_pop, y_train)

            evaluate_model(clfpop, X_train_pop, X_valid_pop, y_train, y_valid)

            save_model_small(clfpop, uid, 'svcpop')

        except Exception as e:
            print(e)
