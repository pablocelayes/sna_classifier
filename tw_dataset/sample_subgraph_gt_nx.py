#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import graph_tool.all as gt

def get_tw_index(ug, v):
    return ug.vp['_graphml_vertex_id'][v]

def is_relevant(v):
    followed_count = v.out_degree()
    followers_count = v.in_degree()
    relevant = followers_count > 40 and followed_count > 40

    return relevant

def get_most_similar_followed(graph, v, amount=None, K=None):
    followed = [f for f in v.out_neighbours() if is_relevant(f)]
    followed_ids = [get_tw_index(graph, f) for f in followed]

    scored = []
    for u in followed:
        u_followed_ids = set([get_tw_index(graph, uf) for uf in u.out_neighbours()])
        common = len(u_followed_ids.intersection(set(followed_ids)))
        total = len(followed_ids) + len(u_followed_ids) - common
        score = common * 1.0 / total
        scored.append((u, score))

    if K is None:
        if amount is not None:
            K = ceil(amount * len(scored))
        else:
            raise ValueError("either N or amount must be passed")
    
    most_similar = sorted(scored, key=lambda (u, s): -s)[:K]
    most_similar = [u for (u, s) in most_similar]

    return most_similar


def sample_subgraph(ug, seeds, K):
    """
        Given a large complete graph, 
        (where all 'followed' relationships between
        its nodes are present) and a set
        of seed nodes, it computes a closure
        graph by adding k most relevant friends while
        possible
    """
    # subgraph = gt.GraphView(ug, directed=True)
    # subgraph = gt.Graph(directed=True)
    subgraph = nx.DiGraph()

    visited = set()
    unvisited = seeds
    while unvisited:
        new_unvisited = set()
        for u in unvisited:
            followed = get_most_similar_followed(ug, u, K=K)
            # TODO: fix this
            u_id = get_tw_index(ug, u)
            followed_ids = [get_tw_index(ug, f) for f in followed]

            subgraph.add_edges_from([(u_id, f_id) for f_id in followed_ids])

            new_unvisited.update(followed)
            visited.add(u)

        new_unvisited = new_unvisited - visited
        unvisited = new_unvisited

        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        print "%d nodes, %d edges" % (n_nodes, n_edges)

    return subgraph

# TODO: restrict universe to layer0 + layer1
if __name__ == '__main__':
    print "Loading universe graph..."
    ug = gt.load_graph('huge_graph.gt')

    print "Loading seeds..."
    seeds = [v for v in ug.vertices() if v.out_degree() == 50]
    print "%d seeds loaded" % len(seeds)

    print "Computing subgraph..."
    subgraph = sample_subgraph(ug, seeds, K=50)
    # subgraph.save('subgraph.gt')
    # nx.write_gpickle(subgraph, 'subgraph.gpickle')
    nx.write_gpickle(subgraph, 'subgraph_gt_nx.gpickle')
