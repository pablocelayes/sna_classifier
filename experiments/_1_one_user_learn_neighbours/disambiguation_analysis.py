#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Process, Manager, Pool
from experiments._1_one_user_learn_neighbours.try_some_users import *
from experiments.utils import *
from experiments.datasets import *
from sklearn.metrics import recall_score
import json


def worker(uid, recalls_train_disamb, recalls_test_disamb, recalls_test_amb, lock):
    """worker function"""
    print "Largamos para %d" % uid
    s = open_session()
    
    user = s.query(User).get(uid)
    clf = load_model(uid)

    X_train, X_test, y_train, y_test = load_or_create_dataframe(uid)

    miss_clf_counts, details = count_doomed_samples(X_train, y_train)

    neighbours = get_neighbourhood(uid)
    error_sources = details

    unames = [x[0][0] for x in error_sources]
    amb_users = {u.username: u for u in neighbours if u.username in unames}
    amb_ids = {u.id for u in amb_users.values()}

    # tweets con un solo 1 en todo X_train y con 1 en y_train
    cands = X_train[(y_train == 1) & (X_train.sum(axis=1) == 1)]

    # y ese en alguno de los amb_users
    amb_tweets = []
    for twid in cands.index:
        t = s.query(Tweet).get(twid)
        if t.author_id in amb_ids:
            amb_tweets.append(t)

    amb_rt_counts = defaultdict(int)
    neighbour_ids = [n.id for n in neighbours]
    for t in amb_tweets:
        for u in t.users_retweeted:
            if u.id not in (neighbour_ids + [user.id]):
                amb_rt_counts[u.id] += 1
        if t.author_id not in (neighbour_ids + [user.id]):
            amb_rt_counts[t.author_id] += 1

    new_neighbours = sorted(amb_rt_counts.items(), key=lambda x: -x[1])
    new_neighbour_ids = [x[0] for x in new_neighbours]
    new_neighbours = [s.query(User).get(nid) for nid in new_neighbour_ids]    

    tweet_ids = X_train.index
    tweets = [s.query(Tweet).get(twid) for twid in tweet_ids]
    X_train_new, _ = extract_features(tweets, new_neighbours, user)
    X_train_new = pd.DataFrame(data=X_train_new, index=tweet_ids, columns=new_neighbour_ids)
    X_train_extended = pd.concat([X_train,X_train_new],axis=1)

    y_true, y_pred = y_test, clf.predict(X_test)

    rs = recall_score(y_true, y_pred)
    lock.acquire()
    recalls_test_amb[uid] = rs
    lock.release()

    # weights for class balancing
    w1 = sum(y_train)/len(y_train)
    w0 = 1 - w1
    sample_weights = np.array([w0 if x==0 else w1 for x in y_train])

    clf = RandomForestClassifier()     
    clf.fit(X_train_extended, y_train, sample_weight=sample_weights)

    y_true, y_pred = y_train, clf.predict(X_train_extended)

    rs = recall_score(y_true, y_pred)
    lock.acquire()
    recalls_train_disamb[uid] = rs
    lock.release()

    #     Extend X_test
    tweet_ids = X_test.index
    tweets = [s.query(Tweet).get(twid) for twid in tweet_ids]
    X_test_new, _ = extract_features(tweets, new_neighbours, user)
    X_test_new = pd.DataFrame(data=X_test_new, index=tweet_ids, columns=new_neighbour_ids)
    X_test_extended = pd.concat([X_test,X_test_new],axis=1)
    y_true, y_pred = y_test, clf.predict(X_test_extended)

    rs = recall_score(y_true, y_pred)
    lock.acquire()
    recalls_test_disamb[uid] = rs
    lock.release()


if __name__ == '__main__':

    with open('recalls_train.json') as f:
        recalls_train = json.load(f)

    with open('recalls_amb.json') as f:
        recalls_amb = json.load(f)

    uids = [int(uid) for uid, recall in recalls_train.items() if recall < 0.9]

    pool = Pool(processes=3)

    manager = Manager()
    recalls_test_disamb = manager.dict()
    recalls_train_disamb = manager.dict()
    recalls_test_amb = manager.dict()
    lock = manager.Lock()

    for uid in uids:
        pool.apply_async(worker, (uid, recalls_train_disamb, recalls_test_disamb, recalls_test_amb, lock))
    pool.close()
    pool.join()

    with open('recalls_train_disamb.json', 'w') as f:
        json.dump(dict(recalls_train_disamb), f)

    with open('recalls_test_disamb.json', 'w') as f:
        json.dump(dict(recalls_test_disamb), f)

    with open('recalls_test_amb.json', 'w') as f:
        json.dump(dict(recalls_test_amb), f)
