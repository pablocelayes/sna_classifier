#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from experiments.relatedness_calculator import finite_katz_measures
from experiments.datasets import *
from experiments.utils import *

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from gensim import matutils

from tw_dataset.dbmodels import *
import pickle, os
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import json

# Low recall users (for neighbor buckets classifier)
# ( <= 80%)
USER = [37226353, "Leandro Deyuanini", 1871]
TEST_USERS = [   
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
        37226353, 
        "Leandro Deyuanini", 
        1871
    ], 
    [
        114582574, 
        "Unión Cívica Radical", 
        689
    ], 
    [
        76684633, 
        "Mario Montoya", 
        129
    ],  
    [
        54987976, 
        "pablorgarcia", 
        126
    ] 
]

def plot_languages(uid):
    dfX, y = load_or_create_dataframe(uid)
    tweetids = dfX.index
    s=open_session()
    tweets = s.query(Tweet).filter(Tweet.id.in_(tweetids)).all()
    langs = [t.lang for t in tweets]
    lang_counts = Counter(langs)
    df = pd.DataFrame.from_dict(lang_counts, orient='index')
    df.plot(kind='bar')


def get_lang_classified_twids(uid, lang):
    """
        get small classified tweet sample
        filtered by language
    """
    fname = '%s_ds_%d.pickle' % (lang, uid)
    if os.path.exists(fname):
        twids, y = pickle.load(open(fname, 'rb'))
    else:
        s=open_session()
        u=s.query(User).get(uid)
        dfX, y = load_or_create_dataframe(u.id)
        tweets = s.query(Tweet).filter(Tweet.id.in_(dfX.index)).all()
        tweets = [t for t in tweets if t.lang==lang]

        alltwids = [t.id for t in tweets]
        rt_twids = [t.id for t in u.timeline if t.id in alltwids]
        nort_twids = alltwids
        [nort_twids.remove(ti) for ti in rt_twids]

        rt_twids = np.random.choice(rt_twids, 100, replace=False)
        # As many of each class
        nort_twids = np.random.choice(nort_twids, 2 * len(rt_twids), replace=False)

        rt_marks = np.ones(len(rt_twids))
        nort_marks = np.zeros(len(nort_twids))

        twids = np.concatenate((rt_twids, nort_twids))
        y = np.concatenate((rt_marks, nort_marks))

        tweets = [t for t in tweets if t.id in twids]
        text = [t.text for t in tweets]
        json.dump(text, open('texts_%d.json' % uid, 'w'))

        pickle.dump((twids, y), open(fname,'wb'))

    return twids, y

if __name__ == '__main__':
    import sys
    n = int(sys.argv[1])
    uid, username, _ = TEST_USERS[n]

    get_lang_classified_twids(uid, 'es')
