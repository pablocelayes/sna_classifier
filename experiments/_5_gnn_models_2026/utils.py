from tw_dataset.settings import DATAFRAMES_FOLDER
import os
import pandas as pd
from os.path import join

import logging

# logging.basicConfig(level=logging.DEBUG)

def load_dataframe_raw(uid, tag="", sparse=False):
    """
        Creates a splitted featurized dataset (social features)
        around a given user.
            target (y): whether uid retweets or not given tweets
            features (X): whether the each user in the 2nd order neighborhood retweeted or not

    """
    logging.debug(f"Processing {uid}")
    if tag:
        tag = f"_{tag}"

    if sparse:
        Xytrain_fname = join(DATAFRAMES_FOLDER, f"raw_sparse/dfXtrain{tag}_{uid}.pickle")
        Xytest_fname = join(DATAFRAMES_FOLDER, f"raw_sparse/dfXtest{tag}_{uid}.pickle")
    else:
        Xytrain_fname = join(DATAFRAMES_FOLDER, f"raw/dfXtrain{tag}_{uid}.pickle")
        Xytest_fname = join(DATAFRAMES_FOLDER, f"raw/dfXtest{tag}_{uid}.pickle")

    exists = False
    if os.path.exists(Xytrain_fname):
        logging.debug("Path exists, trying to load")
        try:
            Xy_train = pd.read_pickle(Xytrain_fname)
            Xy_test = pd.read_pickle(Xytest_fname)
            exists = True
            logging.debug("OK")
        except Exception as e:
            logging.debug(f"Error loading {Xytrain_fname}: {e}")
            pass

    if not exists:
        return None

    feat_col_names = [c for c in Xy_train.columns if c != 'y']
    X_train = Xy_train[feat_col_names]
    y_train = Xy_train["y"]
    X_test = Xy_test[feat_col_names]
    y_test = Xy_test["y"]

    return X_train, X_test, y_train, y_test

def Xy_to_GNN_samples(central_user_id, neighbor_ids, global_inds_map, subgraph_edges, X, y):
    Xy = X.copy()
    Xy["label"] = y
    gnn_samples = []
    for _, r in Xy.iterrows():
        label = r["label"]
        retweeted_ids = [ni for ni in neighbor_ids if r[int(ni)] == 1]

        # Create a sample for the GNN
        sample = {
            "central_user_id": global_inds_map[central_user_id],
            "label": label,
            "neighbor_ids": [global_inds_map[ni] for ni in neighbor_ids],
            "retweeted_ids": [global_inds_map[ri] for ri in retweeted_ids],
            "edge_index": subgraph_edges,
        }
        gnn_samples.append(sample)

    return gnn_samples

def create_gnn_train_val_samples(central_user_id, graph, X_tr, y_tr, X_te, y_te):
    # 1. Get neighbor ids
    neighbor_ids = [str(ni) for ni in X_tr.columns]
    central_user_id = str(central_user_id)
    neighborhood_ids = [central_user_id] + neighbor_ids

    # 2. Get nodes matching those values
    matching_nodes_map = {}
    global_inds_map = {}
    for i, (node, attrs) in enumerate(graph.nodes(data=True)):
        global_inds_map[attrs["twid"]] = i
        if attrs.get("twid") in neighborhood_ids:
            matching_nodes_map[attrs["twid"]] = node

    neighborhood_nodes = [matching_nodes_map[ni] for ni in neighborhood_ids]
    for i, ni in enumerate(neighborhood_nodes):
        graph.nodes[ni]["local_ind"] = i

    # 3. Get all edges between those nodes
    node_set = set(neighborhood_nodes)  # O(1) lookup
    subgraph_edges = [
        (graph.nodes[u]["local_ind"], graph.nodes[v]["local_ind"]) for u, v in graph.edges(neighborhood_nodes)
        if u in node_set and v in node_set
    ]

    # 4. Create GNN samples
    train_gnn_samples = Xy_to_GNN_samples(central_user_id, neighbor_ids, global_inds_map, subgraph_edges, X_tr, y_tr)
    val_gnn_samples = Xy_to_GNN_samples(central_user_id, neighbor_ids, global_inds_map, subgraph_edges, X_te, y_te)


    return train_gnn_samples, val_gnn_samples
