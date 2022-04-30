from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# from datasets import load_or_create_dataset
from experiments.datasets import load_or_create_combined_dataset_small
from experiments.utils import load_ig_graph

# print(__doc__)
import numpy as np
import xgboost as xgb

if __name__ == '__main__':
    # dataset = load_or_create_dataset(37226353)
    # dataset = load_or_create_combined_dataset_small(nmostsimilar=30, nbuckets=20, include_activity_rank=True)
    print("Loading user graph")
    g = load_ig_graph()

    print("Computing centralities")
    centralities = g.pagerank(), g.betweenness(), g.closeness(), g.eigenvector_centrality(), g.eccentricity()
    # centralities = None

    print("Computing bucketized dataset")
    # dataset = load_or_create_combined_dataset_small(g, centralities, nmostsimilar=10, nbuckets=10, include_activity_rank=True)
    dataset = load_or_create_combined_dataset_small(g, centralities, nmostsimilar=10, nbuckets=10, include_activity_rank=True, n_users=4)

    # Fit XGBoost classifier
    X_train, X_test, y_train, y_test = dataset

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(xgb.XGBClassifier(use_label_encoder=False), X_train[:100,:], y_train[:100])
    print(scores)