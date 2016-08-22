#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from gensim import matutils

import pickle, os
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import json

from extractors import EsaFeatureExtractor
from tw_dataset.settings import PREFIX

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
        print("Go to create_samples and create them!")

    return twids, y

# Extract features
def load_or_create_esa_dataset(uid, lang='es'):
    fname = 'esads_%d.npz' % uid
    if os.path.exists(fname):
        z = np.load(open(fname,'rb'))
        X = z['arr_0']
        y = z['arr_1']
    else:
        extractor = EsaFeatureExtractor(PREFIX)
        twids, y = get_lang_classified_twids(uid, lang)
        nfeatures = extractor.get_feature_number()
        nsamples = len(twids)

        X = np.empty((nsamples, nfeatures))

        # texts=s.query(Tweet.text).filter(Tweet.id.in_(twidsX)).all()
        texts = json.load(open("texts_%d.json" % uid))
        for i, text in enumerate(texts):
            sparsefeats = extractor.get_features(text=text)
            fullfeats = sparsefeats.todense()
            X[i, :] = fullfeats[:]

        np.savez(open(fname,'wb'), X, y)

    return X, y

def job_load_or_create_esa_dataset():
    import sys
    n = int(sys.argv[1])
    uid, username, _ = TEST_USERS[n]

    load_or_create_esa_dataset(uid)

if __name__ == '__main__':
    job_load_or_create_esa_dataset()

