#!/usr/bin/env python
# coding: utf-8

from experiments.datasets import *
from os import listdir
from torch_geometric.data import Data
import os
import torch
from os.path import join
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import classification_report, f1_score

# # Super tiny: just for integration test
N_EPOCHS = 4
LIMIT_SAMPLE_USERS = 2
LIMIT_SAMPLE_TWEETS = 20

# Small, but large enough to see if it trains
# N_EPOCHS = 4
# LIMIT_SAMPLE_USERS = 10
# LIMIT_SAMPLE_TWEETS = 200

# Full
# N_EPOCHS = 20
# LIMIT_SAMPLE_USERS = None
# LIMIT_SAMPLE_TWEETS = None

os.environ['TORCH'] = torch.__version__

print("Loading user graph")
g = load_ig_graph()

fname = "../../centralities.pickle"
print("Loading pre-computed centralities")
with open(fname, 'rb') as f:
    centralities = pickle.load(f)

train_samples = {}
test_samples = {}

for fname in listdir(DATAFRAMES_FOLDER):
    df_path = join(DATAFRAMES_FOLDER, fname)
    if fname.count("_") > 1:
        continue
    if fname.startswith("dfXtrain_"):
        samples = train_samples
    elif fname.startswith("dfXtest_"):
        samples = test_samples
    else:
        continue
    user_id = fname.split(".")[0].split("_")[-1]
    if not user_id:
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
    index_centralities = [np.array(m)[index_ig] for m in centralities]

    ### Compute fixed centrality features
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
            if tweet_id in [t.id for t in u.timeline]:
                is_retweeting_col[i] = 1
        is_retweeting_col = np.expand_dims(is_retweeting_col, axis=1)

        x = np.hstack([centralities_matrix, is_central_col, is_retweeting_col])
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(int(tweet_id in [t.id for t in user.timeline]), dtype=torch.long)
        
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

print("Generating pyg Data objects")
print("########Training")
train_dataset = samples_to_data_objects(train_samples)
print("\n########Test")
test_dataset = samples_to_data_objects(test_samples)
torch.manual_seed(12345)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

num_node_features = 7
num_classes = 2

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=5)
print("model")
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        print("Training on batch...")
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        print("loss")
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

print("Training....")
for epoch in range(1, N_EPOCHS + 1):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# ## Save Model
MODEL_PATH = "ggn-model.torch"
torch.save(model, MODEL_PATH)

# ## Evaluate Model
model = torch.load(MODEL_PATH)

preds = []
labels = []
for data in test_loader:
    out = model(data.x, data.edge_index, data.batch)
    preds += out.argmax(dim=1)  # Use the class with highest probability.
    labels += data.y

print(f1_score(labels, preds))
print(classification_report(labels, preds))

