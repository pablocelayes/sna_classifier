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


def load_dataset_large(include_u=True, include_v=True, n_sample=None):
    train_dfs = []
    test_dfs = []
    us = load_user_splits()

    print("Loading user graph")
    g = load_ig_graph()

    print("Loading pre-computed centralities")
    with open("centralities.pickle", 'rb') as f:
        centralities = pickle.load(f)

    # Train users:
    # - train tweets go to train
    # - test tweets go to test
    train_us = []
    if include_u:
        train_us += us["u_train"]
    if include_v:
        train_us += us['au_train']
    for u in train_us:
        X_train, X_test, y_train, y_test = load_or_create_dataframe(u, g, centralities)
        if X_train is None:
            continue
        if sum(y_train) + sum(y_test) < 10:
            print(f"Skipping {u}: too few tweets")
            continue
        X_train["y"] = y_train
        X_test["y"] = y_test
        train_dfs.append(X_train)
        test_dfs.append(X_test)

    # Test users:
    # - all tweets go to test
    test_us = []
    if include_u:
        test_us += us["u_test"]
    if include_v:
        test_us += us['au_test']
    for u in test_us:
        X_train, X_test, y_train, y_test = load_or_create_dataframe(u, g, centralities)
        if X_train is None:
            continue
        if sum(y_train) + sum(y_test) < 10:
            print(f"Skipping {u}: too few tweets")
            continue
        X_train["y"] = y_train
        test_dfs.append(X_train)

        X_test["y"] = y_test
        test_dfs.append(X_test)

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

def count_valid_users(us):
    valid_count = 0
    for u in us:
        try:
            X_train, X_test, y_train, y_test = load_or_create_dataframe(u, g, centralities)
        except Exception:
            continue
        if X_train is None:
            continue
        if sum(y_train) + sum(y_test) < 10:
            print(f"Skipping {u}: too few tweets")
            continue
        valid_count += 1
    return valid_count

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
    print(classification_report(y_true, y_pred, digits=3))

    return clf


def eval_best_model_large():
    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model('best_model_large.xgb')

    X_train, X_test, y_train, y_test = load_dataset_large(n_sample=None, include_u=True, include_v=True)

    print("Scores on test set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    # print classification report
    print(classification_report(y_true, y_pred, digits=3))


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
    print(classification_report(y_true, y_pred, digits=3))


def eval_best_model_general_per_user_u(tag="_large"):
    print("Loading user graph")
    g = load_ig_graph()

    print("Loading pre-computed centralities")
    with open("centralities.pickle", 'rb') as f:
        centralities = pickle.load(f)

    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model(f'best_model{tag}.xgb')

    # compute f1s for each test set
    us=load_user_splits()

    test_splits = ['u_test', 'u_train']
    f1_scores = {s: {} for s in test_splits}
    for split in test_splits:
        for uid in us[split]:
            try:
                X_train, X_test, y_train, y_test = load_or_create_dataframe(uid, g, centralities)
            except Exception:
                continue

            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred, digits=3))

            f1_scores[split][uid] = f1_score(y_true, y_pred)

    with open(f"f1_scores_general_per_user{tag}.json", 'w') as f:
        json.dump(f1_scores, f)

    return f1_scores


def eval_best_model_general_per_user_au(tag="_large"):
    print("Loading user graph")
    g = load_ig_graph()

    print("Loading pre-computed centralities")
    with open("centralities.pickle", 'rb') as f:
        centralities = pickle.load(f)

    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model(f'best_model{tag}.xgb')

    # compute f1s for each test set
    us=load_user_splits()

    test_splits = ['au_test', 'au_train']
    f1_scores = {s: {} for s in test_splits}
    for split in test_splits:
        for uid in us[split]:
            try:
                X_train, X_test, y_train, y_test = load_or_create_dataframe(uid, g, centralities)
                if sum(y_test) + sum(y_train) < 10:
                    print(f"Skipping {u}: too few tweets")
                    continue

            except Exception:
                continue

            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred, digits=3))

            f1_scores[split][uid] = f1_score(y_true, y_pred)

    with open(f"f1_scores_general_per_user{tag}_au.json", 'w') as f:
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
    print(classification_report(y_true, y_pred, digits=3))


def eval_best_model_large_all_U_test_tuits():
    # Load trained best model
    clf = xgb.XGBClassifier()
    clf.load_model('best_model_large.xgb')

    us=load_user_splits()
    u_test_dfs = []
    for u in us['u_train'] + us['u_test']:
        Xy = pd.read_pickle(join(DATAFRAMES_FOLDER, f'dfXtest_{u}.pickle'))
        u_test_dfs.append(Xy)
    Xy_u_test = pd.concat(u_test_dfs)
    y_test = Xy_u_test['y']
    X_test = Xy_u_test
    del X_test['y']
    y_true, y_pred = y_test, clf.predict(X_test)
    # print classification report
    print(classification_report(y_true, y_pred, digits=3))


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
    print(classification_report(y_true, y_pred, digits=3))


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
    print(classification_report(y_true, y_pred, digits=3))


if __name__ == '__main__':
    # train_best_model_large()
    # eval_best_model_general_per_user_au()
    eval_best_model_large()