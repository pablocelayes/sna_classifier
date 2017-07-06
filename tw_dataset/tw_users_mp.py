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

from multiprocessing import Pool, Manager, Process, log_to_stderr, Lock

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

def aux_add_edges_from(edges_list):
    v_edges_list = []
    for (aid, bid) in edges_list:
        aux_add_node(aid)
        aux_add_node(bid)

        va = UID_TO_V[str(aid)]
        vb = UID_TO_V[str(bid)]

        v_edges_list.append((va, vb))

    AUX_GRAPH.add_edge_list(v_edges_list)
    print "AUX: Added %d edges" % len(edges_list)
    print "AUX: Now we have %d vertices and %d edges" % (AUX_GRAPH.num_vertices(), AUX_GRAPH.num_edges())

def aux_save():
    AUX_GRAPH.save('aux_graph.gt')

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
        retries = 0
        while True:
     
            try:
                TW = API_HANDLER.get_connection()
                followed_ids = [str(f_id) for f_id in TW.friends_ids(user_id=user_id)]

                aux_add_node(user_id)
                aux_add_edges_from([(user_id, f_id) for f_id in followed_ids])

                aux_save()

                return followed_ids
            except Exception, e:
                # print e
                if e.message == u'Not authorized.':
                    NOTAUTHORIZED.add(user_id)
                    with open(NOTAUTHORIZED_FNAME, 'w') as f:
                        json.dump(list(NOTAUTHORIZED), f)
                    return None
                else:
                    print "Error for user %s: %s" % (user_id, e.message)
                    retries += 1
                    if retries == 5:
                        print "Gave up retrying for user %s" % user_id
                        return None
                    else:
                        print "waiting..."
                        time.sleep(10)

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



if __name__ == '__main__':
    def worker(fids, lock):
        for user_id in fids:
            if not (user_id in UID_TO_V and UID_TO_V[user_id].out_degree()): # means it was added to graph with its followed ( visited )
                retries = 0
                while True:             
                    try:
                        TW = API_HANDLER.get_connection()
                        followed_ids = [str(f_id) for f_id in TW.friends_ids(user_id=user_id)]

                        lock.acquire()
                        aux_add_node(user_id)
                        aux_add_edges_from([(user_id, f_id) for f_id in followed_ids])
                        aux_save()
                        lock.release()

                    except Exception, e:
                        # print e
                        if e.message == u'Not authorized.':
                            NOTAUTHORIZED.add(user_id)
                            with open(NOTAUTHORIZED_FNAME, 'w') as f:
                                json.dump(list(NOTAUTHORIZED), f)
                            return None
                        else:
                            print "Error for user %s: %s" % (user_id, e.message)
                            retries += 1
                            if retries == 5:
                                print "Gave up retrying for user %s" % user_id
                                return None
                            else:
                                print "waiting..."
                                time.sleep(10)




    K = 50
    
    print "Computing graph..."
    """
        Partiendo de mi usuario,
        voy agregando para cada usuario sus 50 seguidos más similares,
        incluyendo sólo usuarios relevantes ( >40 followers, >40 followed )

        Creamos además un grafo auxiliar con los nodos ya visitados
        (útil para calcular relevancias y similaridades)
    """
    try:
        graph = nx.read_gpickle('graph.gpickle')
    except IOError:
        graph = nx.DiGraph()

    visited = set([x for x in graph.nodes() if graph.out_degree(x)])

    if graph.number_of_nodes():
        unvisited = set([x for x in graph.nodes() if graph.out_degree(x) == 0])
    else:
        unvisited = [USER_DATA['id']]
    
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

            r_followed = [f for f in followed if is_relevant(f)]
            
            # First multiprocess pass of get_followed_ids, to save them to aux graph
            NWORKERS = 6
            p = Pool(NWORKERS)

            batch_size = int(ceil(len(r_followed) * 1.0 / NWORKERS))
            for i in range(NWORKERS):
                fs = r_followed[i * batch_size: (i+1) * batch_size]
                p.apply_async(worker, (fs, lock))            


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
            nx.write_gpickle(graph, 'graph.gpickle')
            aux_save()

            new_unvisited.update(most_similar)

            visited.add(uid)

        new_unvisited = new_unvisited - visited
        unvisited = new_unvisited

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        print "%d nodes, %d edges" % (n_nodes, n_edges)
        
        # save progress
        nx.write_gpickle(graph, 'graph.gpickle')
        aux_save()
        
        with open('failed.json', 'w') as f:
            json.dump(list(failed), f)
