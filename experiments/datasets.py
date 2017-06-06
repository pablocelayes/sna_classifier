#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier

from tw_dataset.dbmodels import *
from tw_dataset.settings import DATASETS_FOLDER, DATAFRAMES_FOLDER
from experiments.utils import *
import pickle, os
import pandas as pd
from os.path import join, exists
from os import remove
import random
from random import sample

import sys
import json

tu_path = "/home/pablo/Proyectos/tesiscomp/tw_dataset/active_with_neighbours.json"

TEST_USERS_ALL = json.load(open(tu_path))

def extract_features(tweets, neighbour_users, own_user):
    '''
        Given tweets and neighbour_users, we extract
        'neighbour activity' features for each tweet

        These are obtained as follows:
            - for each of these users a boolean feature is created
            indicating if the tweet is authored/retweeted by that user
    '''
    nrows = len(tweets)
    nfeats = len(neighbour_users)
    X = np.empty((nrows, nfeats))
    y = np.empty(nrows)

    own_tl_ids = [t.id for t in own_user.timeline]
    for j, u in enumerate(neighbour_users):
        tl_ids = [t.id for t in u.timeline]
        for i, t in enumerate(tweets):
            X[i, j] = 1 if t.id in tl_ids else 0

    for i, t in enumerate(tweets):
        y[i] = 1 if t.id in own_tl_ids else 0

    return X, y


def get_neighbourhood(uid):
    s = open_session()
    user = s.query(User).get(uid)        
    neighbours = get_level2_neighbours(user, s)
    # remove central user from neighbours
    neighbours = [u for u in neighbours if u.id != user.id]

    return neighbours

def load_dataframe(uid):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ys_%d.pickle" % uid)

    X_train = pd.read_pickle(Xtrain_fname)
    X_valid = pd.read_pickle(Xvalid_fname)
    X_test = pd.read_pickle(Xtest_fname)        
    y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_dataframe(uid, max_tweets=10000):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ys_%d.pickle" % uid)

    print("Fetching neighbours...")
    s = open_session()
    user = s.query(User).get(uid)        
    neighbours = get_level2_neighbours(user, s)
    # remove central user from neighbours
    neighbours = [u for u in neighbours if u.id != user.id]
    
    print("Fetching tweets...")
    # Fetch tweet universe (timelines of ownuser and followed)
    followed = get_followed(user, s)
    tweets = set(user.timeline)
    for u in followed:
        tweets.update(u.timeline)

    # exclude tweets from central user or not in Spanish
    tweets = [t for t in tweets if t.author_id != uid and t.lang == 'es']

    print("Extracting features...")
    tweet_ids = [t.id for t in tweets]
    neighbour_ids = [u.id for u in neighbours]
    X, y = extract_features(tweets, neighbours, user)
    s.close()

    X = pd.DataFrame(data=X, index=tweet_ids, columns=neighbour_ids)

    if len(y) > max_tweets:
        neg_inds = [i for i, v in enumerate(y) if v==0]
        pos_inds = [i for i, v in enumerate(y) if v==1]

        n_neg = max_tweets - len(pos_inds)
        neg_inds = sample(neg_inds, n_neg)
        inds = sorted(neg_inds + pos_inds)
        X = X.iloc[inds,:]
        y = y[inds]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.66667, random_state=42)


    X_train.to_pickle(Xtrain_fname)
    X_valid.to_pickle(Xvalid_fname)
    X_test.to_pickle(Xtest_fname)
    pickle.dump((y_train, y_valid, y_test), open(ys_fname, 'wb'))




def create_dataframe_job():
    njob = int(sys.argv[1])
    batch_size = 14
    test_users = TEST_USERS_ALL[1:]
    test_users = test_users[batch_size * njob: batch_size * (njob + 1)]
    for uid in test_users:
        print("Creating dataframe for %d" % uid)
        try:
            load_dataframe(uid)
        except Exception:
            create_dataframe(uid)
    
if __name__ == '__main__':
    create_dataframe_job()