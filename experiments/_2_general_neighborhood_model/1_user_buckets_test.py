from experiments._2_general_neighborhood_model.general_model_evaluation import *
from experiments.datasets import *
u=TEST_USERS_ALL[0]
uid=u[0]

with open("trials_2022-05-24_14:07.pickle", 'rb') as f:
    trials = pickle.load(f)

print("Loading user graph")
g = load_ig_graph()

fname = "centralities.pickle"
print("Loading pre-computed centralities")
with open(fname, 'rb') as f:
    centralities = pickle.load(f)


ds=load_or_create_bucketized_dataset_user(g,centralities, 10, 10, uid)
