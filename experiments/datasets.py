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

tu_path = "/home/pablo/Proyectos/tesiscomp/experiments/_1_one_user_learn_neighbours/active_and_central.json"
TEST_USERS_ALL = json.load(open(tu_path))

def get_active_user(sess):
    rtcounts = {u: len(u.retweets) for u in sess.query(User).all()}
    most_active = sorted(rtcounts.items(), key=lambda x:-x[1])
    
    return most_active


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


def load_or_create_dataset(uid):
    fname = join(DATASETS_FOLDER, "dataset_%d.pickle" % uid)
    if os.path.exists(fname):
        dataset = pickle.load(open(fname, 'rb'))
    else:
        s = open_session()
        user = s.query(User).get(uid)        
        neighbours = get_level2_neighbours(user, s)
        # remove central user from neighbours
        neighbours = [u for u in neighbours if u.id != user.id]
        
        # Fetch tweet universe (timelines of ownuser and neighbours)
        tweets = set(user.timeline)
        for u in neighbours:
            tweets.update(u.timeline)

        # exclude tweets from central user or not in Spanish
        tweets = [t for t in tweets if t.author_id != uid and t.lang == 'es']

        X, y = extract_features(tweets, neighbours, user)
        s.close()
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        dataset = (X_train, X_test, y_train, y_test)

        pickle.dump(dataset, open(fname, 'wb'))


    return dataset


def load_or_create_combined_dataset_small(nbuckets, test_size=0.3):
    fname = join(DATASETS_FOLDER, "dataset_combined_small_b%02d.pickle" % nbuckets)
    if os.path.exists(fname):
        dataset = pickle.load(open(fname, 'rb'))
    else:
        user_border = int((1 - test_size) * len(TEST_USERS))
        train_users, test_users = TEST_USERS[:user_border], TEST_USERS[user_border:]
        
        s = open_session()
        # print("Creating training set based on %d users" % user_border)
        X_train = None
        for uid, username, tweet_count in train_users:
            # print("==================================")
            # print("Loading training set for user %s (id %d)" % (username, uid))
            u_X_train, u_X_test, u_y_train, u_y_test = load_or_create_dataset(uid)

            ds_size, ds_dimension = u_X_train.shape
            # print("Size (#tweets): %d" % ds_size)
            # print("Dimension (#neighbours): %d" % ds_dimension)
            user = s.query(User).get(uid)
            neighbours = get_level2_neighbours(user, s)
            ngids = [str(ng.id) for ng in neighbours]

            u_X = np.vstack((u_X_train, u_X_test))
            u_y = np.hstack((u_y_train, u_y_test))
            u_X = transform_ngfeats_to_bucketfeats(uid, ngids, u_X, nbuckets)

            if X_train is None:
                X_train = u_X
                y_train = u_y
            else:
                X_train = np.vstack((X_train, u_X))
                y_train = np.hstack((y_train, u_y))
        
        ds_size, ds_dimension = X_train.shape
        # print("==================================")    
        # print("Combined training set created.")
        # print("Size (#tweets): %d" % ds_size)
        # print("Dimension (#neighbour buckets): %d" % ds_dimension)

        # print("Creating test set based on remaining users" % user_border)
        X_test = None    
        for uid, username, tweet_count in test_users:
            # print("==================================")
            # print("Loading training set for user %s (id %d)" % (username, uid))
            u_X_train, u_X_test, u_y_train, u_y_test = load_or_create_dataset(uid)

            ds_size, ds_dimension = u_X_train.shape
            # print("Size (#tweets): %d" % ds_size)
            # print("Dimension (#neighbours): %d" % ds_dimension)
            user = s.query(User).get(uid)
            neighbours = get_level2_neighbours(user, s)
            ngids = [str(ng.id) for ng in neighbours]

            u_X = np.vstack((u_X_train, u_X_test))
            u_y = np.hstack((u_y_train, u_y_test))
            u_X = transform_ngfeats_to_bucketfeats(uid, ngids, u_X, nbuckets)

            if X_test is None:
                X_test = u_X
                y_test = u_y
            else:
                X_test = np.vstack((X_train, u_X))
                y_test = np.hstack((y_train, u_y))
        dataset = (X_train, X_test, y_train, y_test)
        pickle.dump(dataset, open(fname, 'wb'))
        s.close()

    return dataset


def load_dataframe(uid):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtest_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ys_%d.pickle" % uid)
    try:
        X_train = pd.read_pickle(Xtrain_fname)
        X_test = pd.read_pickle(Xtest_fname)        
        y_train, y_test = pickle.load(open(ys_fname, 'rb'))
        return X_train, X_test, y_train, y_test
    except Exception as e:
        return None

def repartition_dataframe(uid):
    ds = load_dataframe(uid)

    if ds:
        X_train, X_test, y_train, y_test = ds
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                test_size=0.6667, random_state=42)

        Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d.pickle" % uid)
        Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d.pickle" % uid)
        ys_fname = join(DATAFRAMES_FOLDER, "ysv_%d.pickle" % uid)

        Xtest_fname_old = join(DATAFRAMES_FOLDER, "dfXtest_%d.pickle" % uid)

        # X_train.to_pickle(Xtrain_fname)
        X_valid.to_pickle(Xvalid_fname)
        X_test.to_pickle(Xtest_fname)
        pickle.dump((y_train, y_valid, y_test), open(ys_fname, 'wb'))

        remove(Xtest_fname_old)


def load_validation_dataframe(uid):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ysv_%d.pickle" % uid)

    X_train = pd.read_pickle(Xtrain_fname)
    X_valid = pd.read_pickle(Xvalid_fname)
    X_test = pd.read_pickle(Xtest_fname)        
    y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def reduce_dataset(uid):
    ds = load_validation_dataframe(uid)
    X_train, X_valid, X_test, y_train, y_valid, y_test = ds

    X=pd.concat((X_train,X_valid,X_test))
    y=np.concatenate((y_train,y_valid,y_test))

    if len(y) > 5000:
        neg_inds = [i for i, v in enumerate(y) if v==0]
        pos_inds = [i for i, v in enumerate(y) if v==1]

        n_neg = 5000 - len(pos_inds)
        neg_inds = sample(neg_inds, n_neg)
        inds = sorted(neg_inds + pos_inds)
        X = X.iloc[inds,:]
        y = y[inds]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.66666, random_state=42)

    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d_small.pickle" % uid)
    Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d_small.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d_small.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ysv_%d_small.pickle" % uid)

    X_train.to_pickle(Xtrain_fname)
    X_valid.to_pickle(Xvalid_fname)
    X_test.to_pickle(Xtest_fname)
    pickle.dump((y_train, y_valid, y_test), open(ys_fname, 'wb'))

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_small_validation_dataframe(uid):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d_small.pickle" % uid)
    Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d_small.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d_small.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ysv_%d_small.pickle" % uid)

    X_train = pd.read_pickle(Xtrain_fname)
    X_valid = pd.read_pickle(Xvalid_fname)
    X_test = pd.read_pickle(Xtest_fname)        
    y_train, y_valid, y_test = pickle.load(open(ys_fname, 'rb'))

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def load_or_create_dataframe(uid):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtest_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ys_%d.pickle" % uid)
    exists = False
    if os.path.exists(Xtrain_fname):
        try:
            X_train = pd.read_pickle(Xtrain_fname)
            X_test = pd.read_pickle(Xtest_fname)        
            y_train, y_test = pickle.load(open(ys_fname, 'rb'))
            exists = True
        except Exception as e:
            pass
    
    if not exists:
        s = open_session()
        user = s.query(User).get(uid)        
        neighbours = get_level2_neighbours(user, s)
        # remove central user from neighbours
        neighbours = [u for u in neighbours if u.id != user.id]
        
        # Fetch tweet universe (timelines of ownuser and neighbours)
        tweets = set(user.timeline)
        for u in neighbours:
            tweets.update(u.timeline)

        # exclude tweets from central user or not in Spanish
        tweets = [t for t in tweets if t.author_id != uid and t.lang == 'es']

        tweet_ids = [t.id for t in tweets]
        neighbour_ids = [u.id for u in neighbours]
        X, y = extract_features(tweets, neighbours, user)
        s.close()

        X = pd.DataFrame(data=X, index=tweet_ids, columns=neighbour_ids)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        X_train.to_pickle(Xtrain_fname)
        X_test.to_pickle(Xtest_fname)
        pickle.dump((y_train, y_test), open(ys_fname, 'wb'))

    return X_train, X_test, y_train, y_test






def build_full_graph_dataset():
    # sample users uniformly by katz centrality
    sample_uids = get_full_graph_usersample()

    # Each point consists of a uid and a set of up to 300 tweets
    # in their neighborhood from first 20 days of sample
    datapoints = build_datapoints(sample_uids)

    # Now for each user and all its corresponding datapoints
    # we build
    build_dataframe_from_datapoints()


def get_full_graph_usersample(size=0.7):
    fname = join(DATASETS_FOLDER, 'train_sample_uids.pickle')
    if os.path.exists(fname):
        sample_uids = pickle.load(open(fname, 'rb'))
    else:
        g = load_gt_graph()
        nusers = g.num_vertices()
        nsample = int(nusers * size)
        sorted_twids = get_most_central_twids(N=nusers)
        sample_inds = np.random.choice(nusers, nsample, replace=False)
        sample_uids = [sorted_twids[i] for i in sample_inds]
        
        s = open_session()
        sample_uids = [id for id in sample_uids if s.query(User).get(id)]
        s.close()

        pickle.dump(sample_uids, open(fname, 'wb'))
    
    return sample_uids


def build_datapoints(sample_uids, njob):
    fnames = {}
    datapoints = {}

    for set_type in ["train", "test"]:
        fnames[set_type] = join(DATASETS_FOLDER, 'datapoints_%s_%d.pickle' % (set_type, njob))
        datapoints[set_type] = {}
    
    s = open_session()
    g = load_nx_graph()

    for i, uid in enumerate(sample_uids):
        if i % 50 == 0:
            progress = i * 100 / len(sample_uids)
            print("%.2f %%" % progress)
        user = s.query(User).get(uid)                    
        user_rts = set([t for t in user.timeline if t.author_id != int(uid)])
        
        # Ignore users with less than 30 retweets
        if len(user_rts) < 30:
            continue

        # Fetch tweet universe 
        # (timeline of ownuser + followed)
        neighbours = get_level2_neighbours(user, s)
        tweets_from_neighbours = set()
        for u in neighbours:
            tweets_from_neighbours.update(u.timeline)

        # Keep only RTs that are shared with at least one neighbour
        # ( due to changes in followers it can happen that 
        #   an RT is not shared with any neighbour )
        user_rts = [t for t in user_rts if t in tweets_from_neighbours]
        
        # Make sure that positives (tweets from user) are at
        # least 10% of the sample
        max_followed_tweets = 9 * len(user_rts)
        if len(tweets_from_neighbours) > max_followed_tweets:
            tweets_from_neighbours = sample(tweets_from_neighbours, max_followed_tweets)

        tweets = set()
        tweets.update(user_rts)
        tweets.update(tweets_from_neighbours)

        # Reduce sample to 300 per user
        if len(tweets) > 300:
            tweets = sample(tweets, 300)

        # train/test split
        upper_date_limit = DATE_LOWER_LIMIT + timedelta(days=20)
        datapoints["train"][uid] = [t.id for t in tweets if t.created_at <= upper_date_limit]

        lower_date_limit = DATE_LOWER_LIMIT + timedelta(days=20)
        datapoints["test"][uid] = [t.id for t in tweets if t.created_at > lower_date_limit]
        
    s.close()
    
    for set_type in ["train", "test"]:
        with open(fnames[set_type], 'wb') as f:
            pickle.dump(datapoints[set_type], f)

    return datapoints


def build_datapoints_job():
    njob = int(sys.argv[1])

    sample_uids = get_full_graph_usersample()
    part_size = len(sample_uids) / 8
    uids = sample_uids[part_size * njob: part_size * (njob + 1)]
    build_datapoints(uids, njob)


def combine_datapoints():
    fnames = [join(DATASETS_FOLDER, 'datapoints%d.pickle' % i) for i in range(6)]
    parts = [pd.read_pickle(f) for f in fnames]
    datapoints = pd.concat(parts)
    return datapoints


def build_dataset_from_datapoints(njob, set_type, nbuckets=20, nmostsimilar=30):
    '''
        Given a dictionary of points (users and associated tweets)
        this generates a dataframe of features for those points
        and a vector y of output values. (1 = retweeted, 0 = not retweeted )
    '''
    fname = join(DATASETS_FOLDER, 'datapoints_%s_%d.pickle' % (set_type, njob))
    dp = pickle.load(open(fname, 'rb'))
    s = open_session()
    dfs = []
    ys = []

    for uid in dp:
        user = s.query(User).get(uid)        
        neighbours = get_level2_neighbours(user, s)

        if (len(neighbours) < nmostsimilar + 1):
            continue

        ngids = [n.id for n in neighbours]

        tweets = s.query(Tweet).filter(Tweet.id.in_(dp[uid])).all()
        df_index = [(uid, t.id) for t in tweets]

        Xb, y = extract_features(tweets, neighbours, user)
        X = transform_ngfeats_to_bucketfeats(uid, ngids, Xb, nmostsimilar, nbuckets)

        df = pd.DataFrame(data=X, index=df_index, columns=range(X.shape[1]))
        dfs.append(df)
        ys.append(y)
    s.close()
    dfX = pd.concat(dfs)
    y = np.hstack(ys)

    xfname = join(DATASETS_FOLDER, 'dataset_X_%s_%d.pickle' % (set_type, njob))
    yfname = join(DATASETS_FOLDER, 'dataset_y_%s_%d.pickle' % (set_type, njob))
            
    dfX.to_pickle(xfname)
    pickle.dump(y, open(yfname, 'wb'))
    
    return dfX, y


def load_large_dataset_piece(npiece, set_type):
    xfname = join(DATASETS_FOLDER, 'dataset_X_%s_%d.pickle' % (set_type, npiece))
    yfname = join(DATASETS_FOLDER, 'dataset_y_%s_%d.pickle' % (set_type, npiece))

    dfX = pd.read_pickle(xfname)
    y = pickle.load(open(yfname,'rb'))

    return dfX, y


def load_large_dataset_full(set_type="train"):
    dfs = []
    ys = []
    for npiece in range(8):
        _dfX, _y = load_large_dataset_piece(npiece, set_type)
        dfs.append(_dfX)
        ys.append(_y)
    
    dfX = pd.concat(dfs)
    y = np.hstack(ys)

    return dfX, y


def build_dataset_from_datapoints_job():
    njob = int(sys.argv[1])
    set_type = sys.argv[2]

    build_dataset_from_datapoints(njob=njob, set_type=set_type)


def load_or_create_dataframe_job():
    njob = int(sys.argv[1])
    test_users = TEST_USERS[2 * njob: 2 * njob + 2]
    for uid, _, _ in test_users:
        print("Creating dataframe for %d" % uid)
        load_or_create_dataframe(uid)
    
if __name__ == '__main__':
    # build_datapoints_job()
    build_dataset_from_datapoints_job()