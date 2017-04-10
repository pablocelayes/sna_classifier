#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from utils import json_dump_unicode, json_load_unicode, concatenate
import os
import time

import networkx as nx
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

def get_follower_counts(user_id):
    TW = API_HANDLER.get_connection()
    u = TW.get_user(user_id)
    return u.followers_count

NOTAUTHORIZED_FNAME = "notauthorizedids.pickle"

if os.path.exists(NOTAUTHORIZED_FNAME):
    with open(NOTAUTHORIZED_FNAME, 'rb') as f:
        NOTAUTHORIZED = pickle.load(f)
else:
    NOTAUTHORIZED = set()


def get_neighbour_ids(user_id, graph):
    """
        If user_id is already in graph returns his followed from there.
        Otherwise, it fetches from the API and stores in the graph
    """
    if user_id in graph.nodes() and graph.successors(user_id): # means it was added to graph with its neighbours
        followed = graph.successors(user_id)
        followers = graph.predecessors(user_id)
        return followed, followers
    else:
        retries = 0
        while True:
     
            try:
                TW = API_HANDLER.get_connection()
                followed = TW.friends_ids(user_id=user_id)
                followers = TW.followers_ids(user_id=user_id)

                graph.add_node(user_id)
                graph.add_edges_from([(user_id, f_id) for f_id in followed])
                graph.add_edges_from([(f_id, user_id) for f_id in followers])

                return followed, followers
            except Exception, e:
                # print e
                if e.message == u'Not authorized.':
                    NOTAUTHORIZED.add(user_id)
                    with open(NOTAUTHORIZED_FNAME, 'wb') as f:
                        pickle.dump(NOTAUTHORIZED, f)
                    return []
                else:
                    print "Error for user %d: %s" % (user_id, e.message)
                    retries += 1
                    if retries == 5:
                        print "Gave up retrying for user %d" % user_id
                        return [] 
                    else:
                        print "waiting..."
                        time.sleep(10)

def is_relevant(uid, graph):
    if uid not in RELEVANT:
        followed, followers = get_neighbour_ids(user_id=uid, graph=graph)
        RELEVANT[uid] = len(followed) > 40 and len(followers) > 40    
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
        aux_graph = nx.read_gpickle('aux_graph.gpickle')
    except IOError:
        aux_graph = nx.DiGraph() 

    try:
        graph = nx.read_gpickle('graph.gpickle')
    except IOError:
        graph = nx.DiGraph()

    visited = set([x for x in graph.nodes() if graph.out_degree(x)])

    if graph.number_of_nodes():
        unvisited = set([x for x in graph.nodes() if graph.out_degree(x) == 0])
    else:
        unvisited = [USER_DATA['id']]
    
    while unvisited:
        new_unvisited = set()
        for uid in unvisited:
            followed, _ = get_neighbour_ids(user_id=uid, graph=aux_graph)
            r_followed = [f for f in followed if is_relevant(f, aux_graph)]
            scored = []
            for f in r_followed:
                f_followed, _ = get_neighbour_ids(user_id=f, graph=aux_graph)
                common = len(set(f_followed).intersection(set(followed)))
                total = len(followed) + len(f_followed) - common
                score = common * 1.0 / total
                scored.append((f, score))
            
            most_similar = sorted(scored, key=lambda (u, s): -s)[:K]
            most_similar = [u for (u, s) in most_similar]

            graph.add_edges_from([(uid, f_id) for f_id in most_similar])
            new_unvisited.update(most_similar)

            visited.add(uid)

        new_unvisited = new_unvisited - visited
        unvisited = new_unvisited

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        print "%d nodes, %d edges" % (n_nodes, n_edges)
        
        # save progress
        nx.write_gpickle(graph, 'graph.gpickle')
        nx.write_gpickle(graph, 'aux_graph.gpickle')


    return graph, aux_graph

if __name__ == '__main__':
    print "Computing graph..."
    graph, aux_graph = create_graphs(K=50)
