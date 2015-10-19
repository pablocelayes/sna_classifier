#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn import datasets
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
    """
        Given tweets and neighbour_users, we extract
        'neighbour activity' features for each tweet

        These are obtained as follows:
            - for each of these users a boolean feature is created
            indicating if the tweet is authored/retweeted by that user
    """
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


def load_or_create_dataframe(uid=USER_ID):
    fname = join(DATAFRAMES_FOLDER, "dfX_%d.pickle" % uid)
    yfname = join(DATAFRAMES_FOLDER, "y_%d.pickle" % uid)
    if os.path.exists(fname):
        dataframe = pd.read_pickle(fname)
        y = pickle.load(open(yfname, 'rb'))
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

        dataframe = pd.DataFrame(data=X, index=tweet_ids, columns=neighbour_ids)
        dataframe.to_pickle(fname)
        pickle.dump(y, open(yfname, 'wb'))

    return dataframe, y


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


def build_datapoints(sample_uids, njob=None):
    if njob is not None:
        fname = join(DATASETS_FOLDER, 'datapoints%d.pickle' % njob)
    else:
        fname = join(DATASETS_FOLDER, 'datapoints.pickle')
    if os.path.exists(fname):
        datapoints = pd.read_pickle(fname)
    else:
        datapoints = pd.DataFrame(index=sample_uids, columns=range(300))
        s = open_session()
        for i, uid in enumerate(sample_uids):
            if i % 50 == 0:
                progress = i * 100 / len(sample_uids)
                print("%.2f %%" % progress)
            user = s.query(User).get(uid)        
            neighbours = get_level2_neighbours(user, s)
            
            # Fetch tweet universe (timelines of ownuser and neighbours)
            tweets = set(user.timeline)
            for u in neighbours:
                tweets.update(u.timeline)
            tweets = list(tweets)
            
            upper_date_limit = DATE_LOWER_LIMIT + timedelta(days=20)
            twids = [t.id for t in tweets if t.created_at < upper_date_limit]
            if len(twids) > 300:
                twids = np.random.choice(twids, 300, replace=False)

            for j, twid in enumerate(twids):
                datapoints.loc[uid, j] = twid
        s.close()
        datapoints.to_pickle(fname)

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


def build_dataset_from_datapoints(dp=None, njob=None, nbuckets=20):
    """
        Given a dataframe of points (users and associated tweets)
        this generates a dataframe of features for those points
        and a vector y of output values. (retweeted or not)
    """
    if njob is not None:
        fname = join(DATASETS_FOLDER, 'datapoints%d.pickle' % njob)
        dp = pd.read_pickle(fname)
    s = open_session()
    dfs = []
    ys = []

    for uid in dp.index:
        user = s.query(User).get(uid)        
        neighbours = get_level2_neighbours(user, s)
        if not neighbours:
            continue
        ngids = [n.id for n in neighbours]
        tweets = s.query(Tweet).filter(Tweet.id.in_(dp.loc[uid])).all()
        df_index = [(uid, t.id) for t in tweets]

        X, y = extract_features(tweets, neighbours, user)
        X = transform_ngfeats_to_bucketfeats(uid, ngids, X, nbuckets)

        df = pd.DataFrame(data=X, index=df_index, columns=range(nbuckets))
        dfs.append(df)
        ys.append(y)
    s.close()
    dfX = pd.concat(dfs)
    y = np.hstack(ys)

    xfname = join(DATASETS_FOLDER, 'large_X%d.pickle' % njob)
    dfX.to_pickle(xfname)

    yfname = join(DATASETS_FOLDER, 'large_y%d.pickle' % njob)
    pickle.dump(y, open(yfname, 'wb'))
    
    return dfX, y


def build_dataset_from_datapoints_job():
    njob = int(sys.argv[1])
    build_dataset_from_datapoints(njob=njob)


def load_or_create_dataframe_job():
    njob = int(sys.argv[1])
    test_users = TEST_USERS[2 * njob: 2 * njob + 2]
    for uid, _, _ in test_users:
        print("Creating dataframe for %d" % uid)
        load_or_create_dataframe(uid)
    
if __name__ == '__main__':
    # build_datapoints_job()
    build_dataset_from_datapoints_job()