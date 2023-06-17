#!/usr/bin/env python
# coding: utf-8
import sys

import numpy as np

from experiments.datasets import *
from experiments.utils import *
from os import listdir
from torch_geometric.data import Data
from torch_geometric.utils import scatter
import os
import torch
from os.path import join
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, avg_pool_x
from sklearn.metrics import classification_report, f1_score
from scipy import sparse
from time import time

# TRAIN = False
TRAIN = True
N_BATCHES_LOG_FREQUENCY = 1
N_HIDDEN_CHANNELS = 128
MODEL_PATH = "ggn-model.torch"

def get_onehot(user_id):
    v = np.zeros(len(g.vs))
    ind = g.vs['twid'].index(str(user_id))
    v[ind] = 1
    return v

# ### Implementación más sencilla
# 
# - sin embeddings
# - sólo el vecindario de u
# - features: is_u, is_rt, centralidades (misma info que paper anterior)
def create_pyg_data_objects(tweet_ids, user_id, neighbour_users):
    '''
        Given tweets, central user and neighbour_users,
        we extract a pytorchgeometric Data representation
        of the 'neighbour activity' for each tweet
        
        nodes = user + retweeting neighbors + other neighbors
        features:
            centralities
            is_central_user
            is_retweeting
    '''
    s = open_session()
    user = s.query(User).get(user_id)

    ### Compute index mappings (igraph, twid, row_id)

    # print("# Filter centralities to cover only ngids")
    # This is the fixed order we will keep for nodes in x,
    # for all the generated datasamples for this central user
    user_id = user.id
    neighbour_ids = [u.id for u in neighbour_users]    
    index_users = [user] + neighbour_users
    index_twid = [user_id] + neighbour_ids    
    index_twid_to_igraph_map = {int(l): i for (i,l) in enumerate(g.vs["twid"]) if int(l) in index_twid}
    index_ig = [index_twid_to_igraph_map[l] for l in index_twid]

    ### Compute one-hot encodings
    # row = np.arange(len(index_ig))
    # col = np.array(index_ig)
    # data = np.ones(len(index_ig))
    # onehots = sparse.csr_matrix((data, (row, col)), shape=(len(index_ig), len(g.vs)))

    onehots = np.zeros((len(index_ig), len(g.vs)))
    for i, j in enumerate(index_ig):
        onehots[i,j] = 1

    ### Compute fixed centrality features
    index_centralities = [np.array(m)[index_ig] for m in centralities]
    centralities_matrix = np.vstack(index_centralities).transpose()

    is_central_col = np.zeros(len(index_users))
    is_central_col[0] = 1
    is_central_col = np.expand_dims(is_central_col, axis=1)

    ## Compute edges
    # select edgeds within subgraph
    edges = g.es.select(_within=index_ig)
    edges_igraph = [e.tuple for e in edges]

    # now we need to map them to row indices in the x feature matrix
    # igraph_ind -> tw_ind -> row_ind

    igraph_ind_to_tw_ind_map = {ig: tw for (tw, ig) in index_twid_to_igraph_map.items()}
    tw_ind_to_row_ind_map = {tw: i for (i, tw) in enumerate(index_twid)}
    def map_ig_ind_to_row_ind(ig):
        return tw_ind_to_row_ind_map[igraph_ind_to_tw_ind_map[ig]]

    def map_edge(e):
        a, b = e
        return (map_ig_ind_to_row_ind(a), map_ig_ind_to_row_ind(b))
    
    edge_index = torch.tensor([map_edge(e) for e in edges_igraph])
    edge_index = edge_index.transpose(0,1)

    ### Compute retweeting features for each example
    data_objects = []
    for tweet_id in tweet_ids:
        is_retweeting_col = np.zeros(len(index_users))
        for i, u in enumerate(index_users):
            # retweeting features for all users except central (otherwise we would be leaking here)
            if i > 0 and tweet_id in [t.id for t in u.timeline]:
                is_retweeting_col[i] = 1
        is_retweeting_col = np.expand_dims(is_retweeting_col, axis=1)

        # x = sparse.hstack([onehots, centralities_matrix, is_central_col, is_retweeting_col], format="csr")
        x = np.hstack([onehots, centralities_matrix, is_central_col, is_retweeting_col])
        # import ipdb; ipdb.set_trace()
        # x = x.todense()
        x = torch.tensor(x, dtype=torch.float32)
        # row, col = x.nonzero()
        # x = torch.sparse_csr_tensor(
        #     torch.tensor(row, dtype=torch.float32),
        #     torch.tensor(col, dtype=torch.float32),
        #     torch.tensor(x.data, dtype = torch.float32),
        #     size=x.shape
        # )
        # x = x.to_dense()
        y = torch.tensor([int(tweet_id in [t.id for t in user.timeline])], dtype=torch.float32)
        
        d = Data(x=x, edge_index=edge_index, y=y)

        data_objects.append(d)
        
    return data_objects

def samples_to_data_objects(samples):
    data_objects = []
    s = open_session()

    for user_id, tweet_ids in list(samples.items()):
        print(user_id)
        user = s.query(User).get(user_id)
        neighbours = get_level2_neighbours(user, s)
        # remove central user from neighbours
        neighbour_users = [u for u in neighbours if u.id != user.id]
        data_objects += create_pyg_data_objects(tweet_ids, user_id, neighbour_users)
    return data_objects

def compute_label_weights(train_loader):
    label_counts = [0, 0]
    for data in train_loader:
        n_samples = len(data.y.numpy())
        n_pos = data.y.numpy().sum()
        label_counts[1] += n_pos
        label_counts[0] += n_samples - n_pos

    n_labels = sum(label_counts)
    return torch.tensor([n_labels / c for c in label_counts], dtype=torch.float32)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels * 3, num_classes)
        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index, batch):
        # last column is mask for retweeting users
        # second to last column is mask for centrals
        cluster = x[:, -1] + 2 * x[:, -2]

        # TODO: maybe exclude centrality metrics from the convolutional layers?

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        # pool central, retweeting and others separately and concatenate
        # x_user
        # x_retweeters
        # x_others

        # x, _ = avg_pool_x(cluster, x, batch, batch_size=BATCH_SIZE)
        cluster_batch = (batch * 3 + cluster).long()
        xc = scatter(x, cluster_batch, dim=0, reduce='mean')
        batch_size = len(batch.unique())
        hidden_channels = xc.shape[-1]

        x = xc.reshape(batch_size, 3 * hidden_channels)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        return self.sigmoid(x)

def train(model, train_loader, criterion, class_weights):
    model.train()
    i = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        target = data.y
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        out = torch.squeeze(out)

        intermediate_losses = criterion(out, target)  # Per sample,
                                                      # now we need to weight them and add them
        weights = class_weights[1] * target + class_weights[0] * (1 - target)
        loss = torch.mean(weights * intermediate_losses)

        if (i % N_BATCHES_LOG_FREQUENCY) == 0:
            print(f"Loss: {loss}")
        i += 1

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def evaluate(loader, print_results=True):
    preds = []
    labels = []
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        preds += out.argmax(dim=1)  # Use the class with highest probability.
        labels += data.y

    f1 = f1_score(labels, preds)

    if print_results:
        print(f1)
        print(classification_report(labels, preds))

    return f1

#### Running configurations ######
def get_running_config(config_name):
    if config_name == "tiny":
        # Super tiny: just for integration test
        N_EPOCHS = 4
        LIMIT_SAMPLE_USERS = 1
        LIMIT_SAMPLE_TWEETS = 200
        ONLY_ACTIVE_USERS = True
        N_BATCHES_LOG_FREQUENCY = 1
        BATCH_SIZE = 10
    elif config_name == "small":
        # Small, but large enough to have some more positive examples
        N_EPOCHS = 4
        LIMIT_SAMPLE_USERS = 4
        LIMIT_SAMPLE_TWEETS = 500
        ONLY_ACTIVE_USERS = True
        N_BATCHES_LOG_FREQUENCY = 1
        BATCH_SIZE = 100
    elif config_name == "mid":
        # Mid-size, hopefully large enough to learn to predict some positive examples
        N_EPOCHS = 20
        LIMIT_SAMPLE_USERS = 2
        LIMIT_SAMPLE_TWEETS = None
        ONLY_ACTIVE_USERS = True
        N_BATCHES_LOG_FREQUENCY = 1
        BATCH_SIZE = 1024
    elif config_name == "full":
        # Full
        N_EPOCHS = 20
        LIMIT_SAMPLE_USERS = None
        LIMIT_SAMPLE_TWEETS = None
        ONLY_ACTIVE_USERS = False
        N_BATCHES_LOG_FREQUENCY = 1
        BATCH_SIZE = 1024
    print(f"""
        Config: {config_name}
        ----------------------
        N_EPOCHS: {N_EPOCHS},
        LIMIT_SAMPLE_USERS: {LIMIT_SAMPLE_USERS},
        LIMIT_SAMPLE_TWEETS: {LIMIT_SAMPLE_TWEETS},
        ONLY_ACTIVE_USERS {ONLY_ACTIVE_USERS},
        N_BATCHES_LOG_FREQUENCY: {N_BATCHES_LOG_FREQUENCY}
        BATCH_SIZE: {BATCH_SIZE}
    """)

    return (
        N_EPOCHS,
        LIMIT_SAMPLE_USERS,
        LIMIT_SAMPLE_TWEETS,
        ONLY_ACTIVE_USERS,
        N_BATCHES_LOG_FREQUENCY,
        BATCH_SIZE
    )

if __name__ == '__main__':
    config_name = sys.argv[1]
    (
        N_EPOCHS,
        LIMIT_SAMPLE_USERS,
        LIMIT_SAMPLE_TWEETS,
        ONLY_ACTIVE_USERS,
        N_BATCHES_LOG_FREQUENCY,
        BATCH_SIZE
    ) = get_running_config(config_name)

    os.environ['TORCH'] = torch.__version__

    print("Loading user graph...")
    g = load_ig_graph()

    fname = "../../centralities.pickle"
    print("Loading pre-computed centralities")
    with open(fname, 'rb') as f:
        centralities = pickle.load(f)

    train_samples = {}
    test_samples = {}

    # We start with active and central users
    us = load_user_splits()

    for fname in sorted(listdir(DATAFRAMES_FOLDER)):
        df_path = join(DATAFRAMES_FOLDER, fname)
        if fname.count("_") > 1:
            continue
        if fname.startswith("dfXtrain_"):
            samples = train_samples
        elif fname.startswith("dfXtest_"):
            samples = test_samples
        else:
            continue
        if len(samples) == LIMIT_SAMPLE_USERS:
            continue
        user_id = fname.split(".")[0].split("_")[-1]
        # TODO: remove this, now it is filtering only active users
        if ONLY_ACTIVE_USERS and not int(user_id) in us['u_train']:
            continue
        Xy = pd.read_pickle(df_path)
        tweets = Xy.index.values.tolist()
        if LIMIT_SAMPLE_TWEETS:
            tweets = tweets[:LIMIT_SAMPLE_TWEETS]
        samples[user_id] = tweets

    if LIMIT_SAMPLE_USERS:
        train_samples = dict(list(train_samples.items())[:LIMIT_SAMPLE_USERS])
        test_samples = dict(list(test_samples.items())[:LIMIT_SAMPLE_USERS])

    print("train and test users:")
    print(len(train_samples))
    print(len(test_samples))

    print("Generating pyg Data objects")
    print("########Training")
    train_dataset = samples_to_data_objects(train_samples)
    from random import shuffle

    shuffle(train_dataset)

    print("\n########Test")
    test_dataset = samples_to_data_objects(test_samples)
    torch.manual_seed(12345)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    import ipdb; ipdb.set_trace()


    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    model = GCN(hidden_channels=N_HIDDEN_CHANNELS,
                num_node_features = 5596, # TODO: compute from graph
                num_classes=1)
    print("model")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print("Computing label weights")
    class_weights = compute_label_weights(train_loader)
    print(class_weights)
    # criterion = torch.nn.CrossEntropyLoss(weight=weight)
    # criterion = torch.nn.BCELoss(weight=weight)
    criterion = torch.nn.BCELoss(reduction='none')

    if TRAIN:
        print("Training....")
        for epoch in range(1, N_EPOCHS + 1):
            # train
            epoch_start = time()
            train(model, train_loader, criterion, class_weights)
            train_time = time() - epoch_start

            # eval
            eval_start = time()
            # train_acc = test(train_loader)
            # test_acc = test(test_loader)
            f1_test = evaluate(test_loader, print_results=False)
            eval_time = time() - eval_start

            print(f'Epoch: {epoch:03d}, ' +
                  # 'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, ' +
                  f' Test F1: {f1_test:.2f}, Train time: {int(train_time):d} secs, Eval time {int(eval_time):d} secs')

        # ## Save Model
        # TODO: fix pickle error
        # _pickle.PicklingError: Can't pickle <class '__main__.GCN'>: attribute lookup GCN on __main__ failed
        # torch.save(model, MODEL_PATH)

    # ## Evaluate Model
    # print("Loading saved model....")
    # model = torch.load(MODEL_PATH)

    print("Evaluation on training data")
    evaluate(train_loader)

    print("Evaluation on test data")
    evaluate(test_loader)

    import ipdb; ipdb.set_trace()

