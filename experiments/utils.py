import graph_tool.all as gt
import networkx as nx
from tw_dataset.dbmodels import *
from tw_dataset.settings import PROJECT_PATH, GT_GRAPH_PATH, NX_GRAPH_PATH
from experiments.relatedness import finite_katz_measures
from collections import defaultdict

from os.path import join
import numpy as np

def load_nx_graph():
    return nx.read_gpickle(NX_GRAPH_PATH)

def load_gt_graph():
    return gt.load_graph(GT_GRAPH_PATH)

def get_twitter_id(g, v):
    return g.vertex_properties['_graphml_vertex_id'][v]

def get_vertex_by_twid(g, twid):
    for v in g.vertices():
        if get_twitter_id(g, v) == str(twid):
            return v

def get_central_and_followed():
    g = gt.load_graph(GT_GRAPH_PATH)
    katz_centrality = gt.katz(g)
    central_id = katz_centrality.get_array().argmax()
    central = g.vertex(central_id)
    central_twid = get_twitter_id(g, central_id)
    followed_twids = [get_twitter_id(g, n) for n in central.out_neighbours()]

    return central_twid, followed_twids

def get_most_central_twids(N=100):
    g = gt.load_graph(GT_GRAPH_PATH)
    katzc=gt.katz(g)
    katzc_array = katzc.get_array()
    katzc_sorted = sorted(enumerate(katzc_array), key=lambda (i, v):v)
    most_central = [id for (id, c) in katzc_sorted][:N]
    most_central_twids = [get_twitter_id(g,id) for id in most_central]
    
    return most_central_twids

def get_retweets(user_ids):
    s = open_session()
    retweets = set()
    for uid in user_ids:
        user = s.query(User).get(uid)
        if user:
            # TODO: check why some are None
            retweets.update(user.retweets)

    return retweets

def get_followed(user, session, g):
    uid = str(user.id)
    followed = set(g.successors(uid))
    followed_users = [session.query(User).get(twid) for twid in followed]

    # Remove None elements and own user (TODO: see why this happens)
    followed_users = [u for u in followed_users if u and u.id != user.id]
    
    return followed_users

def get_followers(user, session, g):
    uid = str(user.id)
    followers = set(g.predecessors(uid))
    follower_users = [session.query(User).get(twid) for twid in followers]

    # Remove None elements and own user (TODO: see why this happens)
    follower_users = [u for u in follower_users if u and u.id != user.id]
    
    return follower_users


def get_level2_neighbours(user, session):
    """
        An ordered list of up to level-2 neighbours is created
        (followed and their followed)
    """
    # g = load_gt_graph()
    g = load_nx_graph()
    # v = get_vertex_by_twid(g, user.id)
    uid = str(user.id)

    # neighbourhood = set(v.out_neighbours())
    neighbourhood = set(g.successors(uid))

    for nid in g.successors(uid):
        neighbourhood.update(g.successors(nid))

    # neighbourhood_twids = [get_twitter_id(g, n) for n in neighbourhood]
    # neighbour_users = [session.query(User).get(twid) for twid in neighbourhood_twids]
    neighbour_users = [session.query(User).get(twid) for twid in neighbourhood]

    # Remove None elements and own user (TODO: see why this happens)
    neighbour_users = [u for u in neighbour_users if u and u.id != user.id]
    
    return neighbour_users

# Feature transformations
def transform_ngfeats_to_bucketfeats(uid, ngids, Xfeats, nmostsimilar=30, nbuckets=20):
    '''
        We transform neighbour retweet features into a bucketed
        fixed-length format as follows:
           * neighbours are sorted by katz similarity
           * the 30 most similar stay with individual columns
           * the remaining are grouped in 20 buckets of similarity level
    '''
    # Calculate katz similarities
    g = load_nx_graph()
    fkatz_sims = finite_katz_measures(g, str(uid), K=10, alpha=0.2)
    ngs_fkatz = {i: fkatz_sims[str(i)] for i in ngids}
    sorted_ngs_fkatz = sorted(ngs_fkatz.items(), key=lambda (i, s): s)
    twid_to_colind = { twid: colind for colind, twid in enumerate(ngids)}

    # Create first buckets with most similar users
    if nmostsimilar:
        most_similar = sorted_ngs_fkatz[-nmostsimilar:]
        most_similar_colinds = [[twid_to_colind[twid]] for twid, s in most_similar]
        sorted_ngs_fkatz = sorted_ngs_fkatz[:-nmostsimilar]
    else:
        most_similar_colinds = []

    # Group the remaining ones in nbuckets
    ngs_logfkatz = [(x, np.log(k)) for (x,k) in sorted_ngs_fkatz]
    lfk_range = [ngs_logfkatz[i][1] for i in [0,-1]]
    
    minfk, maxfk = lfk_range
    dfk = maxfk - minfk
    
    step = dfk / nbuckets
    endpoints = [minfk + i * step for i in range(1, nbuckets)]

    groups_colinds = [[] for i in range(nbuckets)]
    group_i = 0
    for (twid, logfkatz) in ngs_logfkatz:
        if group_i < nbuckets - 1 and logfkatz > endpoints[group_i]:
            group_i += 1
        colind = twid_to_colind[twid]
        groups_colinds[group_i].append(colind)

    # Filter data frame and sum
    bucket_colinds = most_similar_colinds + groups_colinds
    bucket_columns = [Xfeats[:, colinds].sum(axis=1) for colinds in bucket_colinds]
    grXfeats = np.column_stack(bucket_columns)

    return grXfeats

def get_unique_rows(a):
    """
        Given a numpy array, it returns
        a new array containing only its unique rows,
        with no repetitions
    """
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx, counts = np.unique(b, return_index=True, return_counts=True)

    return a[idx], counts

def vector_to_neighbour_list(v, neighbours):
    ns = [neighbours[i] for i in range(len(v)) if v[i]]
    return [n.username for n in ns]

def count_doomed_samples(X, y):
    """
        Given a training dataset, we find
        its contradictions and compute the minimum
        possible number of missclassified samples
        by summing the sizes of smallest groups among
        all pairs of contradictory groups
    """
    s = open_session()
    neighbours = [s.query(User).get(uid) for uid in X.columns]
    Xy = merge_Xy(X, y)
    Xy_u, counts = get_unique_rows(Xy)

    miss_clf_counts = defaultdict(float)
    sample_counts={}
    details = []
    for i, r in enumerate(Xy_u):
        s = r[:-1].tostring()
        y = r[-1]
        c = counts[i]
        if s in sample_counts:
            if c > sample_counts[s]:
                miss_clf_counts[int(1-y)] += sample_counts[s]
            elif c < sample_counts[s]:
                miss_clf_counts[int(y)] += c
            # TODO: decidir q hacer en este caso
            # else:
            #     miss_clf_counts[0] += c * 0.5
            #     miss_clf_counts[1] += c * 0.5
            v = r[:-1]
            cs = {y: c, 1-y: sample_counts[s]}
            nl = vector_to_neighbour_list(v, neighbours)
            details.append((nl, cs))
        else:
            sample_counts[s] = c

    # The cost of an inconsistency is the minimum number of missclasified
    # samples
    details = sorted(details, key=lambda (x, y): -min(y.values()))

    return miss_clf_counts, details

def merge_Xy(X, y):
    Xy = np.hstack((X, np.zeros((X.shape[0],1))))
    Xy[:,-1] = y

    return Xy
