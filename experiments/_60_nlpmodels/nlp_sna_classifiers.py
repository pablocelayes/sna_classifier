#!/usr/bin/env python
# -*- coding: utf-8 -*-
from create_lda_datasets import *
# from experiments._1_one_user_learn_neighbours.try_some_users import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report

from experiments._1_one_user_learn_neighbours.classifiers import model_select_svc

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))


MODELS_FOLDER = "/media/pablo/data/Tesis/models/old/"

def save_model_small(clf, user_id, model_type, feat_space=''):
    model_path = join(MODELS_FOLDER, "%s_%d_small%s.pickle" % (model_type, user_id, feat_space))
    joblib.dump(clf, model_path)

def load_model_small(user_id, model_type, feat_space=''):
    model_path = join(MODELS_FOLDER, "%s_%d_small%s.pickle" % (model_type, user_id, feat_space))
    clf = joblib.load(model_path)
    return clf

if __name__ == '__main__':
    f1s = load_nlp_selected_users()
    
    for uid, f1 in f1s:
        uid = int(uid)
        print "Processing %d ( f1 %.2f %%)" % (uid, 100 * f1)

        X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(uid)
        X_train_lda, X_valid_lda, X_test_lda, y_train, y_valid, y_test = load_lda_dataset(uid)

        X_train_combined = sp.hstack((X_train, X_train_lda))
        X_valid_combined = sp.hstack((X_valid, X_valid_lda))
        X_test_combined = sp.hstack((X_test, X_test_lda))

        ds_comb = (X_train_combined, X_valid_combined, y_train, y_valid)
        comb_clf = model_select_svc(ds_comb)

        save_model_small(comb_clf, uid, 'svc', 'comb')

        print "Results on old (pure social) model"
        sna_clf = load_model_small(uid, 'svc')
        evaluate_model(sna_clf, X_train, X_valid, y_train, y_valid)