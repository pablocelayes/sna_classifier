#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

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

from experiments._65_linkedarticles.extractors import EsaFeatureExtractor
from tw_dataset.settings import PREFIX


def load_dataframe(uid=USER_ID):
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

    return X_train, X_test, y_train, y_test

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

