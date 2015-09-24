#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx

def is_relevant(graph, user_id):
    # TODO: Check if large graph is excluding relevants, to 
    #  see if this is necessary 
    followed_count = graph.in_degree(user_id)
    followers_count = graph.out_degree(user_id)
    relevant = followers_count > 40 and followed_count > 40

    return relevant

def get_most_similar_followed(graph, user_id, amount=None, K=None):
    followed_ids = graph.successors(user_id)
    followed_ids = [f_id for f_id in followed_ids if is_relevant(graph, f_id)]

    scored = []
    for u_id in followed_ids:
        u_followed_ids = set(graph.successors(u_id))
        common = len(u_followed_ids.intersection(set(followed_ids)))
        total = len(followed_ids) + len(u_followed_ids) - common
        score = common * 1.0 / total
        scored.append((u_id, score))

    if K is None:
        if amount is not None:
            K = ceil(amount * len(scored))
        else:
            raise ValueError("either N or amount must be passed")
    
    most_similar = sorted(scored, key=lambda (u, s): -s)[:K]
    most_similar = [u for (u, s) in most_similar]

    return most_similar


def sample_subgraph(universe_graph, seed_ids, K):
    """
        Given a large complete graph, 
        (where all 'followed' relationships between
        its nodes are present) and a set
        of seed nodes, it computes a closure
        graph by adding k most relevant friends while
        possible
    """
    subgraph = nx.DiGraph()
    visited = set()
    unvisited = seed_ids
    while unvisited:
        new_unvisited = set()
        for u_id in unvisited:
            followed = get_most_similar_followed(universe_graph, u_id, K=K)
            subgraph.add_edges_from([(u_id, f_id) for f_id in followed])

            new_unvisited.update(followed)
            visited.add(u_id)

        new_unvisited = new_unvisited - visited
        unvisited = new_unvisited

        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        print "%d nodes, %d edges" % (n_nodes, n_edges)


if __name__ == '__main__':
    print "Loading universe graph..."
    universe_graph = nx.read_gpickle('universe_graph.gpickle')

    print "Loading seed ids..."
    fg1 = nx.read_gpickle('filtered_graph1.gpickle')
    seed_ids = [n for n in fg1.nodes() if fg1.out_degree(n)]
    print "%d seed ids loaded" % len(seed_ids)

    print "Computing subgraph..."
    subgraph = sample_subgraph(universe_graph, seed_ids, K=50)
    nx.write_gpickle(subgraph, 'subgraph.gpickle')