#!/usr/bin/env python
# -*- coding: utf-8 -*-
from experiments._1_one_user_learn_neighbours.try_some_users import *
from experiments.utils import *
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = load_or_create_dataset(126527644)
# Loading dataset for user Décimo Doctor (id 126527644)
# Dataset loaded.
# Size (#tweets): 322971
# Dimension (#neighbours): 814

Xy_train = merge_Xy(X_train,y_train)
Xy_test = merge_Xy(X_test,y_test)
Xy = np.vstack((Xy_train, Xy_test))

train_perc = 0.7

Xy_u, _ = get_unique_rows(Xy)

cut = int(0.7 * len(Xy_u))

Xy_train_u = Xy_u[:cut]
Xy_test_u = Xy_u[cut:]
X_train_u, y_train_u = Xy_train_u[:,:-1],  Xy_train_u[:,-1]
X_test_u, y_test_u = Xy_test_u[:,:-1],  Xy_test_u[:,-1]

clf = RandomForestClassifier()

w1 = sum(y_train_u)/len(y_train_u)
w0 = 1 - w1
sample_weights = np.array([w0 if x==0 else w1 for x in y_train_u])

clf.fit(X_train_u, y_train_u, sample_weight=sample_weights)

y_true, y_pred = y_train_u, clf.predict(X_train_u)
print("Scores on train set.\n")
print(classification_report(y_true, y_pred))

y_true, y_pred = y_test_u, clf.predict(X_test_u)
print("Scores on test set.\n")
print(classification_report(y_true, y_pred))
