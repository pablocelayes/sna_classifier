#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Methods for cleaning up, navigating and reducing fetched
    Twitter graph
"""
from tw_users import GRAPH, get_most_similar_followed
import pickle
import networkx as nx
import os

def extend_followed_graph(outer_layer_ids, level):
    """
        Given a graph and the ids of its outer layer,
        it extends it with one extra step in the followed
        relation.

        This is meant to be applied up to a certain number
        of steps. The outer layer are the nodes that were
        seen for the first time in the previous step

        Level 0 are just a set of selected relevant users.
        After that, level N + 1 is the extension of level N
        by one more step in the followed relation
    """
    fname_current = 'filtered_graph%d.gpickle' % level     
    if os.path.exists(fname_current):     # resume
        graph = nx.read_gpickle(fname_current)
        with open('filtered_layer%d.pickle' % level, 'rb') as fl:
            new_outer_layer = pickle.load(fl)
    else:
        # start from previous
        fname_previous = 'filtered_graph%d.gpickle' % (level - 1)
        graph = nx.read_gpickle(fname_previous)
        new_outer_layer = set()

    seen = set([x[0] for x in graph.edges()])
    unvisited = outer_layer_ids - seen
    for i, u_id in enumerate(unvisited):
        print "Adding filtered edges for %d" % u_id
        followed = get_most_similar_followed(user_id=u_id, N=50)
        graph.add_edges_from([(u_id, f_id) for f_id in followed])
        
        new_nodes = [f_id for f_id in followed if graph.out_degree(f_id) == 0]
        new_outer_layer.update(new_nodes)
        
        if i % 10 == 0:
            # Save only ocassionally
            print "Saving snapshots..."
            nx.write_gpickle(graph, fname_current)
            nx.write_gpickle(GRAPH, 'big_graph.gpickle')
            with open('filtered_layer%d.pickle' % level, 'wb') as fl:
                pickle.dump(new_outer_layer, fl)

    return graph, new_outer_layer


def compute_extended_graphs():
    with open('filtered_layer0.pickle','rb') as f:
        outer_layer_ids = pickle.load(f)

    for level in [1, 2]:
        graph, outer_layer_ids = extend_followed_graph(outer_layer_ids, level)


if __name__ == '__main__':
    compute_extended_graphs()
