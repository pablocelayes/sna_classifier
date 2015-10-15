#!/usr/local/bin python
# -*- coding: utf-8 -*-
__author__ = 'Pablo Celayes'

"""
    Implementation of various known
    metrics for node relatedness in an
    undirected graph
"""
from collections import defaultdict
import pandas as pd
from os.path import exists

def katz_measure(G, node_a, node_b):
    raise NotImplementedError

def finite_katz_measures(G, node, alpha=0.5, K=4):
    """
        Given a a graph and a node belonging to it,
        this computes a truncated version of Katz 
        relatedness measure for all relevant nodes
        (being at distance at most K from node)

        Parameters
        ----------
        G : graph
            A NetworkX graph

        node: node
            A node in G

        alpha: float ∊ (0, 1)
            path-length attenuation factor 

        K: int
            maximum path length

        Returns
        -------
        measures : dictionary
           Dictionary of nodes with finite Katz relatedness to
           node as the value.  
    """
    # This will hold, as the path_length increases,
    # a dictionary of nodes reachable with current path_lenght
    # having the number of paths for its value
    path_length = 0
    reachable = {node: 1}

    measures = defaultdict(float)
    while path_length < K:
        path_length += 1
        # Update reachable counts
        new_reachable = defaultdict(int)    

        for n, path_count in reachable.items():
            for nn in G.neighbors(n):
                new_reachable[nn] += path_count
        reachable = new_reachable

        # Add to measures
        for n, path_count in reachable.items():
            measures[n] += path_count * (alpha ** (path_length - 1))

    return measures


def commute_time(graph, node_a, node_b):
    raise NotImplementedError


def hitting_time(graph, node_a, node_b):
    raise NotImplementedError

def all_finite_katz_measures(G, alpha=0.2, K=10):
    """
        Given a a graph, this computes a truncated version of Katz 
        relatedness measure between each pair of nodes
        being at distance at most K

        Parameters
        ----------
        G : graph
            A NetworkX graph

        alpha: float ∊ (0, 1)
            path-length attenuation factor 

        K: int
            maximum path length

        Returns
        -------
        measures :
            a data frame that in position [id_a, id_b]
            contains relatedness of node_b to node_a  
    """
    fname = "fkatz_measures.pickle"
    if os.path.exists(fname):
        measures = pd.read_pickle(fname)
    else:
        nodes = G.nodes()
        measures = pd.DataFrame(0, index=nodes, columns=nodes)

        for node in nodes:
            path_length = 0
            reachable = {node: 1}

            while path_length < K:
                path_length += 1
                # Update reachable counts
                new_reachable = defaultdict(int)    

                for n, path_count in reachable.items():
                    for nn in G.neighbors(n):
                        new_reachable[nn] += path_count
                reachable = new_reachable

                # Add to measures
                for n, path_count in reachable.items():
                    measures.at[node, n] += path_count * (alpha ** (path_length - 1))
        measures.to_pickle(fname)

    return measures
