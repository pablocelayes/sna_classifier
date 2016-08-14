#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier

from tw_dataset.dbmodels import *
from tw_dataset.settings import DATASETS_FOLDER, DATAFRAMES_FOLDER
from experiments.utils import *
import pickle, os
import pandas as pd
from os.path import join
import random
import sys


TEST_USERS = [
    [
        203030351, 
        "Gustavo Gomez", 
        1021
    ],    
    [
        228252737, 
        "LAWRENCE JPD ARABIA ", 
        2523
    ], 
    [
        142800528, 
        "@elprofesionalll", 
        2139
    ], 
    [
        18623370, 
        "RadioLaRioja.com.ar", 
        1909
    ], 
    [
        37226353, 
        "Leandro Deyuanini", 
        1871
    ], 
    # NOTE: We leave this one out because it produces a huge dataset
    # which causes an error trying to save it to pickle
    # [
    #     195412602, 
    #     "res non verba ", 
    #     1856
    # ],
    [
        101047126, 
        "Javier A Lewkowicz", 
        764
    ], 
    [
        126527644, 
        "Décimo Doctor", 
        714
    ], 
    [
        114582574, 
        "Unión Cívica Radical", 
        689
    ], 
    [
        144310601, 
        "sexpoerotica", 
        688
    ], 
    [
        20611326, 
        "Juan Muñoz", 
        678
    ],
        [
        1566310694, 
        "Evelyn", 
        129
    ], 
    [
        59857143, 
        "Mundo D ", 
        129
    ], 
    [
        76684633, 
        "Mario Montoya", 
        129
    ], 
    [
        85123028, 
        "Mariano", 
        127
    ], 
    [
        54987976, 
        "pablorgarcia", 
        126
    ] 
]

USER_ID = 203030351
# [
#     203030351, 
#     "Gustavo Gomez", 
#     1021
# ], 

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


def load_or_create_dataset(uid=USER_ID):
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
        tweets = list(tweets)

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


def load_or_create_dataframe(uid=USER_ID):
    Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
    Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtest_%d.pickle" % uid)
    ys_fname = join(DATAFRAMES_FOLDER, "ys_%d.pickle" % uid)
    if os.path.exists(Xtrain_fname):
        X_train = pd.read_pickle(Xtrain_fname)
        X_test = pd.read_pickle(Xtest_fname)        
        y_train, y_test = pickle.load(open(ys_fname, 'rb'))
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
        tweets = list(tweets)

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
        user_tweets = set(user.timeline)
        
        # Ignore users with less than 30 tweets
        if len(user_tweets) < 30:
            continue

        # Fetch tweet universe 
        # (timeline of ownuser + followed)
        followed = get_followed(user, s, g)
        tweets_from_followed = set()
        for u in followed:
            tweets_from_followed.update(u.timeline)
        
        # Make sure that positives (tweets from user) are at
        # least 10% of the sample
        max_followed_tweets = 9 * len(user_tweets)
        if len(tweets_from_followed) > max_followed_tweets:
            tweets_from_followed = random.sample(tweets_from_followed, max_followed_tweets)

        tweets = set()
        tweets.update(user_tweets)
        tweets.update(tweets_from_followed)

        # Reduce sample to 300 per user
        if len(tweets) > 300:
            tweets = random.sample(tweets, 300)

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


def build_dataset_from_datapoints(njob, set_type, nbuckets=20):
    '''
        Given a dataframe of points (users and associated tweets)
        this generates a dataframe of features for those points
        and a vector y of output values. (retweeted or not)
    '''
    fname = join(DATASETS_FOLDER, 'datapoints_%s_%d.pickle' % (set_type, njob))
    dp = pickle.load(open(fname, 'rb'))
    s = open_session()
    dfs = []
    ys = []

    for uid in dp:
        user = s.query(User).get(uid)        
        neighbours = get_level2_neighbours(user, s)
        if not neighbours:
            continue
        ngids = [n.id for n in neighbours]
        tweets = s.query(Tweet).filter(Tweet.id.in_(dp[uid])).all()
        df_index = [(uid, t.id) for t in tweets]

        Xb, y = extract_features(tweets, neighbours, user)
        X = transform_ngfeats_to_bucketfeats(uid, ngids, Xb, nbuckets)

        df = pd.DataFrame(data=X, index=df_index, columns=range(nbuckets))
        dfs.append(df)
        ys.append(y)
    s.close()
    dfX = pd.concat(dfs)
    y = np.hstack(ys)

    if set_type == 'train':
        xfname = join(DATASETS_FOLDER, 'large_X%d.pickle' % njob)
        yfname = join(DATASETS_FOLDER, 'large_y%d.pickle' % njob)
    elif set_type == "testtime":
        xfname = join(DATASETS_FOLDER, 'large_Xtt_%d.pickle' % njob)
        yfname = join(DATASETS_FOLDER, 'large_ytt_%d.pickle' % njob)
            
    dfX.to_pickle(xfname)
    pickle.dump(y, open(yfname, 'wb'))
    
    return dfX, y


def load_large_dataset_piece(npiece, set_type):
    if set_type == "train":
        xfname = join(DATASETS_FOLDER, 'large_X%d.pickle' % npiece)
        yfname = join(DATASETS_FOLDER, 'large_y%d.pickle' % npiece)
    elif set_type == "testtime":
        xfname = join(DATASETS_FOLDER, 'large_Xtt_%d.pickle' % npiece)
        yfname = join(DATASETS_FOLDER, 'large_ytt_%d.pickle' % npiece)


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