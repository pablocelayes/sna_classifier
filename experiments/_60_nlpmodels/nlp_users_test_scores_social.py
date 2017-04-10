#!/usr/bin/env python
# -*- coding: utf-8 -*-
from create_twlda_datasets_15_and_fit_classifiers import *
# from experiments._1_one_user_learn_neighbours.try_some_users import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report

from experiments._1_one_user_learn_neighbours.classifiers import model_select_svc
from sklearn.preprocessing import StandardScaler

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))


MODELS_FOLDER = "/media/pablo/data/Tesis/models/"

def save_model_small(clf, user_id, model_type, feat_space='', n_topics=None):
    n_topics_str = 't%d' % n_topics if n_topics else ''
    fname = '_'.join(x for x in [model_type, str(user_id), 'small', feat_space, n_topics_str] if x)
    model_path = join(MODELS_FOLDER, "%s.pickle" % fname)
    joblib.dump(clf, model_path)

def load_model_small(user_id, model_type, feat_space='', n_topics=None):
    n_topics_str = 't%d' % n_topics if n_topics else ''
    fname = '_'.join(x for x in [model_type, str(user_id), 'small', feat_space, n_topics_str] if x)
    model_path = join(MODELS_FOLDER, "%s.pickle" % fname)

    clf = joblib.load(model_path)
    return clf

def scale(X_train, X_test):
    train_size = X_train.shape[0]
    X = np.concatenate((X_train.todense(), X_test.todense()))
    X = StandardScaler().fit_transform(X)
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]

    return X_train, X_test

if __name__ == '__main__':
    f1s = load_nlp_selected_users()
    
    for uid, f1 in f1s:
        uid = int(uid)
        print "==============================" 
        print "Processing %d ( f1 %.2f %%)" % (uid, 100 * f1)

        X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(uid)
        
        sna_clf = load_model_small(uid, 'svc')
        evaluate_model(sna_clf, X_train, X_test, y_train, y_test)