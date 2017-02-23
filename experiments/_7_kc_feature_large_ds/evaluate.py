#!/usr/bin/env python
# -*- coding: utf-8 -*-
from experiments.datasets import *
from experiments.utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def add_centrality_feature(dfX):
    g = gt.load_graph(GT_GRAPH_PATH)
    katzc = gt.katz(g)
    katzc_array = katzc.get_array()
    uids = list(set([uid for (uid, _) in dfX.index]))
    kcs = {uid: katzc[get_vertex_by_twid(g,uid)] for uid in uids}
    dfX['kc'] = [kcs[uid] for (uid, _) in dfX.index]


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weight = np.array([w0 if x==0 else w1 for x in y_train]) 

    clf.fit(X_train, y_train, sample_weight=sample_weight)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    y_true, y_pred = y_train, clf.predict(X_train)
    print(classification_report(y_true, y_pred))

    print("Scores on test set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    dfX, y = load_large_dataset_full()
    add_centrality_feature(dfX)
    X_train, X_test, y_train, y_test = train_test_split(dfX, y, test_size=0.3, random_state=42)

    print("===== Evaluating with KC feature ========")
    clf = RandomForestClassifier()
    # clf = SVC(kernel="rbf", gamma=1, C=0.1, class_weight='auto') <---- ¡para atrás!

    train_and_evaluate(clf, X_train, X_test, y_train, y_test)

    print("===== Evaluating without KC feature ========")
    del X_train['kc']
    del X_test['kc']
    clf = RandomForestClassifier()
    # clf = SVC(kernel="rbf", gamma=1, C=0.1, class_weight='auto')

    train_and_evaluate(clf, X_train, X_test, y_train, y_test)



