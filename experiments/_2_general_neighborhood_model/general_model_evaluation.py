import pickle
import json
from experiments.datasets import load_or_create_combined_dataset_small, load_user_splits, DATAFRAMES_FOLDER, load_or_create_dataframe
from experiments.datasets import TEST_USERS_ALL, load_or_create_bucketized_dataset_user, N_BUCKETS, N_MOST_SIMILAR
from experiments.utils import load_ig_graph
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
import pandas as pd

import os
from os.path import join
# N_USERS = 4
N_USERS = None


def load_best_params():
    with open("trials_2022-05-24_14:07.pickle", 'rb') as f:
        trials = pickle.load(f)
    return {k: v[0] for (k, v) in trials.best_trial['misc']['vals'].items()}


def load_dataset(best_params):
    print("Loading user graph")
    g = load_ig_graph()

    fname = "centralities.pickle"
    print("Loading pre-computed centralities")
    with open(fname, 'rb') as f:
        centralities = pickle.load(f)

    print("Computing bucketized dataset")
    dataset = load_or_create_combined_dataset_small(g, centralities,
                                                    n_users=N_USERS,
                                                    nmostsimilar=best_params['nmostsimilar'],
                                                    nbuckets=best_params['nbuckets'],
                                                    include_activity_rank=True)

    return dataset


def load_dataset_large(n_sample=None):
    train_dfs = []
    test_dfs = []
    us = load_user_splits()

    # Train users:
    # - train tweets go to train
    # - test tweets go to test
    for u in us['u_train'] + us['au_train']:
        fname = join(DATAFRAMES_FOLDER, f"dfXtrain_{u}.pickle")
        if not os.path.exists(fname):
            continue
        Xy = pd.read_pickle(fname)
        if Xy["y"].sum() < 10:
            print(f"Skipping {u}: too few tweets")
            continue
        train_dfs.append(Xy)

        Xyt = pd.read_pickle(join(DATAFRAMES_FOLDER, f"dfXtest_{u}.pickle"))
        test_dfs.append(Xyt)

    # Test users:
    # - all tweets go to test
    for u in us['u_test'] + us['au_test']:
        fname = join(DATAFRAMES_FOLDER, f"dfXtrain_{u}.pickle")
        if not os.path.exists(fname):
            continue
        Xy = pd.read_pickle(fname)
        if Xy["y"].sum() < 10:
            print(f"Skipping {u}: too few tweets")
            continue
        test_dfs.append(Xy)

        Xyt = pd.read_pickle(join(DATAFRAMES_FOLDER, f"dfXtest_{u}.pickle"))
        test_dfs.append(Xyt)

    Xy_train = pd.concat(train_dfs)
    Xy_test = pd.concat(test_dfs)
    if n_sample:
        Xy_train = Xy_train.sample(n_sample)

    feat_col_names = [f"#{i + 1}" for i in range(N_MOST_SIMILAR)] + [f"b{b + 1}" for b in range(N_BUCKETS)]
    X_train = Xy_train[feat_col_names]
    y_train = Xy_train["y"]
    X_test = Xy_test[feat_col_names]
    y_test = Xy_test["y"]
    print("Train")
    print(X_train.shape)
    print("Test")
    print(X_test.shape)

    return X_train, X_test, y_train, y_test


def train_best_model():
    # Load best parameters
    best_params = load_best_params()

    # Get dataset for best params
    dataset = load_dataset(best_params)
    X_train, X_test, y_train, y_test = dataset

    del best_params['nmostsimilar']
    del best_params['nbuckets']

    # Train model
    clf = xgb.XGBClassifier(use_label_encoder=False, **best_params)
    clf.fit(X_train, y_train)

    # Save model
    clf.save_model('best_model.xgb')


def train_best_model_large():
    # Load best parameters
    best_params = load_best_params()
    del best_params['nmostsimilar']
    del best_params['nbuckets']

    # Get dataset for best params
    X_train, X_test, y_train, y_test = load_dataset_large(n_sample=None)

    # Train model
    # best_params["n_estimators"] = 10
    print(best_params)

    clf = xgb.XGBClassifier(use_label_encoder=False, verbosity=2, **best_params)
    clf.fit(X_train, y_train)

    # Save model
    clf.save_model('best_model_large.xgb')

    print("Scores on test set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    # print classification report
    print(classification_report(y_true, y_pred))

    return clf


def eval_best_model_general():
    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model('best_model.xgb')

    # Load combined test set
    best_params = load_best_params()
    dataset = load_dataset(best_params)
    _, X_test, _, y_test = dataset

    # Compute combined f1 score
    print("Scores on test set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    # print classification report
    print(classification_report(y_true, y_pred))


def eval_best_model_general_per_user(tag="_large"):
    print("Loading user graph")
    g = load_ig_graph()

    print("Loading pre-computed centralities")
    with open("centralities.pickle", 'rb') as f:
        centralities = pickle.load(f)

    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model(f'best_model{tag}.xgb')

    # Load test sets for test users
    users = TEST_USERS_ALL
    user_border = int(0.7 * len(users))
    test_users = users[user_border:]

    best_params = load_best_params()

    # compute f1s for each test set
    us=load_user_splits()

    test_splits = ['u_test', 'u_train']
    f1_scores = {s: {} for s in test_splits}
    for split in test_splits:
        for uid in us[split]:
    # for uid, username, tweet_count in test_users:
            try:
                X_train, X_test, y_train, y_test = load_or_create_dataframe(uid, g, centralities)
            except Exception:
                continue

            # X_train, X_test, y_train, y_test = load_or_create_bucketized_dataset_user(g, centralities,
            #                                                                           nmostsimilar=best_params['nmostsimilar'],
            #                                                                           nbuckets=best_params['nbuckets'],
            #                                                                           uid=uid)
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))

            f1_scores[split][uid] = f1_score(y_true, y_pred)

    with open(f"f1_scores_general_per_user{tag}.json", 'w') as f:
        json.dump(f1_scores, f)

    return f1_scores


def eval_best_model_large_test_U():
    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model('best_model_large.xgb')

    us=load_user_splits()
    u_test_dfs = []
    for u in us['u_test']:
        Xy = pd.read_pickle(join(DATAFRAMES_FOLDER, f'dfXtest_{u}.pickle'))
        u_test_dfs.append(Xy)
    Xy_u_test = pd.concat(u_test_dfs)
    y_test = Xy_u_test['y']
    X_test = Xy_u_test
    del X_test['y']
    y_true, y_pred = y_test, clf.predict(X_test)
    # print classification report
    print(classification_report(y_true, y_pred))


def eval_best_model_large_train_U_test_tuits():
    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model('best_model_large.xgb')

    us=load_user_splits()
    u_test_dfs = []
    for u in us['u_train']:
        Xy = pd.read_pickle(join(DATAFRAMES_FOLDER, f'dfXtest_{u}.pickle'))
        u_test_dfs.append(Xy)
    Xy_u_test = pd.concat(u_test_dfs)
    y_test = Xy_u_test['y']
    X_test = Xy_u_test
    del X_test['y']
    y_true, y_pred = y_test, clf.predict(X_test)
    # print classification report
    print(classification_report(y_true, y_pred))


def eval_best_model_large_train_AU_test_tuits():
    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model('best_model_large.xgb')

    us=load_user_splits()
    u_test_dfs = []
    for u in us['u_train'] + us['au_train']:
        fname = join(DATAFRAMES_FOLDER, f'dfXtest_{u}.pickle')
        if not os.path.exists(fname):
            continue
        Xy = pd.read_pickle(fname)
        if Xy["y"].sum() < 10:
            print(f"Skipping {u}: too few tweets")
            continue
        u_test_dfs.append(Xy)

    Xy_u_test = pd.concat(u_test_dfs)
    y_test = Xy_u_test['y']
    X_test = Xy_u_test
    del X_test['y']
    y_true, y_pred = y_test, clf.predict(X_test)
    # print classification report
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    train_best_model_large()