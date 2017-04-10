#!/usr/bin/env python
# -*- coding: utf-8 -*-
from create_lda_datasets import *
# from experiments._1_one_user_learn_neighbours.try_some_users import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report

from experiments._1_one_user_learn_neighbours.classifiers import model_select_svc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    
    start_next = False

    params = {
        186998381: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        263741712: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        213462875: {'kernel': 'rbf', 'C': 0.01, 'gamma': 0.1, 'class_weight': 'balanced'},
        295873172: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        173348160: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        67699758: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        200302248: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        356464236: {'kernel': 'rbf', 'C': 0.01, 'gamma': 0.1, 'class_weight': 'balanced'},
        283762641: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        1383498264: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        143685841: {'kernel': 'rbf', 'C': 0.1, 'gamma': 0.1, 'class_weight': 'balanced'},
        157523545: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        151540361: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        166394396: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        186595337: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        196478764: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        152769475: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        125363239: {'kernel': 'rbf', 'C': 0.1, 'gamma': 0.1, 'class_weight': 'balanced'},
        173550390: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        107176039: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        256674213: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        91919237: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        127163121: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        199982337: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        151058509: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        152821325: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        128386349: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        18145859: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        1311735576: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': 'balanced'},
        17620434: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        229480408: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        214986628: {'kernel': 'rbf', 'C': 0.1, 'gamma': 0.1, 'class_weight': 'balanced'},
        59857143: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        118547936: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        158116807: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        263872477: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None},
        208575740: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1, 'class_weight': None}
    }

    for uid, f1 in f1s:
        uid = int(uid)

        print "Processing %d ( f1 %.2f %%)" % (uid, 100 * f1)

        X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(uid)
        X_train_lda, X_valid_lda, X_test_lda, y_train, y_valid, y_test = load_lda_dataset(uid, ntopics=100)

        X_train_combined = sp.hstack((X_train, X_train_lda))
        X_valid_combined = sp.hstack((X_valid, X_valid_lda))
        X_test_combined = sp.hstack((X_test, X_test_lda))

        X_train_combined, X_valid_combined = scale(X_train_combined, X_valid_combined)
        
        try:
            clf = load_model_small(uid, 'svc', 'comb', n_topics=100)
        except Exception as e:

            ds_comb = (X_train_combined, X_valid_combined, y_train, y_valid)

            comb_clf = SVC(**params[uid])

            comb_clf.fit(X_train_combined, y_train) 

            save_model_small(comb_clf, uid, 'svc', 'comb', n_topics=100)
