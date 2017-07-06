#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from utils import json_dump_unicode, json_load_unicode, concatenate
import os
import time

import networkx as nx
import graph_tool.all as gt
import pickle, json
from datetime import timedelta, datetime
from twitter_api import API_HANDLER
from settings import USER_DATA
from math import ceil


RELEVANT_FNAME = "relevantdict.json"

if os.path.exists(RELEVANT_FNAME):
    with open(RELEVANT_FNAME, 'r') as f:
        RELEVANT = json.load(f)
else:
    RELEVANT = {}

NOTAUTHORIZED_FNAME = "notauthorizedids.json"

if os.path.exists(NOTAUTHORIZED_FNAME):
    with open(NOTAUTHORIZED_FNAME, 'r') as f:
        NOTAUTHORIZED = set(json.load(f))
else:
    NOTAUTHORIZED = set()

try:
    AUX_GRAPH = gt.load_graph('aux_graph.gt')
    V_TO_UID = AUX_GRAPH.vp['_graphml_vertex_id']
    UID_TO_V = {V_TO_UID[v]: v for v in AUX_GRAPH.vertices()}
except IOError:
    AUX_GRAPH = gt.Graph(directed=True)

def aux_add_node(user_id):
    user_id = str(user_id)
    if user_id not in UID_TO_V:
        print "Adding %s" % user_id
        new_v = AUX_GRAPH.add_vertex()
        V_TO_UID[new_v] = user_id
        UID_TO_V[user_id] = new_v


def get_followed_ids(user_id):
    """
        If user_id is already in graph returns his followed from there.
        Otherwise, it fetches from the API and stores in the graph
    """
    user_id = str(user_id)
    if user_id in UID_TO_V and UID_TO_V[user_id].out_degree(): # means it was added to graph with its followed ( visited )
        followed_ids = [V_TO_UID[v] for v in UID_TO_V[user_id].out_neighbours()]
        return followed_ids
    else:
        return []

def is_relevant(uid):
    uid = str(uid)
    if uid not in RELEVANT:
        TW = API_HANDLER.get_connection()        
        u = TW.get_user(uid)
        RELEVANT[uid] = u.friends_count > 40 and u.followers_count > 40    
        with open(RELEVANT_FNAME, 'w') as f:
            json.dump(RELEVANT, f)

    return RELEVANT[uid]

def create_graphs(K=50):
    """
        Partiendo de mi usuario,
        voy agregando para cada usuario sus 50 seguidos más similares,
        incluyendo sólo usuarios relevantes ( >40 followers, >40 followed )

        Creamos además un grafo auxiliar con los nodos ya visitados
        (útil para calcular relevancias y similaridades)
    """
    try:
        graph = nx.read_gpickle('graph2.gpickle')
    except IOError:
        graph = nx.DiGraph()

    visited = set([x for x in graph.nodes() if graph.out_degree(x)])

    if graph.number_of_nodes():
        unvisited = set([x for x in graph.nodes() if graph.out_degree(x) == 0])
    else:
        unvisited = [str(USER_DATA['id'])]
    
    try:
        failed = set(json.load(open('failed.json')))
    except IOError:
        failed = set()

    while unvisited:
        new_unvisited = set()
        for uid in unvisited:
            followed = get_followed_ids(user_id=uid)

            if followed is None:
                failed.add(int(uid))
                continue

            # r_followed = [f for f in followed if is_relevant(f)]
            r_followed = followed # All nodes in universe are assumed relevant
            scored = []
            for f in r_followed:
                f_followed = get_followed_ids(user_id=f)
                if f_followed is None:
                    failed.add(int(f))
                    continue

                common = len(set(f_followed).intersection(set(followed)))
                total = len(followed) + len(f_followed) - common
                score = common * 1.0 / total
                scored.append((f, score))
            
            most_similar = sorted(scored, key=lambda (u, s): -s)[:K]
            most_similar = [u for (u, s) in most_similar]

            graph.add_edges_from([(uid, f_id) for f_id in most_similar])
            nx.write_gpickle(graph, 'graph2.gpickle')
            

            new_unvisited.update(most_similar)

            visited.add(uid)

        new_unvisited = new_unvisited - visited
        unvisited = new_unvisited

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        print "%d nodes, %d edges" % (n_nodes, n_edges)
        
        # save progress
        nx.write_gpickle(graph, 'graph2.gpickle')
        
        
        # with open('failed.json', 'w') as f:
        #     json.dump(list(failed), f)


    return graph, AUX_GRAPH

if __name__ == '__main__':
    print "Computing graph..."
    graph, AUX_GRAPH = create_graphs(K=50)
