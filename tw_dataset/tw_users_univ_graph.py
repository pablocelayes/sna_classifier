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


RELEVANT_FNAME = "relevantdict_u.json"

if os.path.exists(RELEVANT_FNAME):
    with open(RELEVANT_FNAME, 'r') as f:
        RELEVANT = json.load(f)
else:
    RELEVANT = {}

NOTAUTHORIZED_FNAME = "notauthorizedids_u.json"

if os.path.exists(NOTAUTHORIZED_FNAME):
    with open(NOTAUTHORIZED_FNAME, 'r') as f:
        NOTAUTHORIZED = set(json.load(f))
else:
    NOTAUTHORIZED = set()

try:
    UNIV_GRAPH = gt.load_graph('univ_graph.gt')
    V_TO_UID = UNIV_GRAPH.vp['_graphml_vertex_id']
    UID_TO_V = {V_TO_UID[v]: v for v in UNIV_GRAPH.vertices()}
except IOError:
    UNIV_GRAPH = gt.Graph(directed=True)

def univ_add_node(user_id):
    user_id = str(user_id)
    if user_id not in UID_TO_V:
        print "Adding %s" % user_id
        new_v = UNIV_GRAPH.add_vertex()
        V_TO_UID[new_v] = user_id
        UID_TO_V[user_id] = new_v

def univ_add_edges_from(edges_list):
    v_edges_list = []
    for (aid, bid) in edges_list:
        univ_add_node(aid)
        univ_add_node(bid)

        va = UID_TO_V[str(aid)]
        vb = UID_TO_V[str(bid)]

        v_edges_list.append((va, vb))

    UNIV_GRAPH.add_edge_list(v_edges_list)
    print "AUX: Added %d edges" % len(edges_list)
    print "AUX: Now we have %d vertices and %d edges" % (UNIV_GRAPH.num_vertices(), UNIV_GRAPH.num_edges())
    # import ipdb; ipdb.set_trace()

def univ_save():
    UNIV_GRAPH.save('univ_graph.gt')

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

                univ_add_node(user_id)
                univ_add_edges_from([(user_id, f_id) for f_id in followed_ids])

                univ_save()

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
        retries = 0
        while True:
            try:
                TW = API_HANDLER.get_connection()        
                u = TW.get_user(uid)
                RELEVANT[uid] = u.friends_count > 40 and u.followers_count > 40
                break
            except Exception as e:
                print "Error for user %s: %s" % (uid, e.message)
                retries += 1
                if retries == 5:
                    print "Gave up retrying for user %s" % uid
                    RELEVANT[uid] = False 
                    break
                else:
                    print "waiting..."
                    time.sleep(10)

        with open(RELEVANT_FNAME, 'w') as f:
            json.dump(RELEVANT, f)

    return RELEVANT[uid]

def create_univ_graph(K):
    """
        Partiendo de mi usuario,
        voy agregando para cada usuario sus K seguidos más similares,
        incluyendo sólo usuarios relevantes ( >40 followers, >40 followed )

        Creamos además un grafo auxiliar con los nodos ya visitados
        (útil para calcular relevancias y similaridades)
    """

    completed_level = 0
    visited = set()

    unvisited = [str(USER_DATA['id'])]
    
    try:
        failed = set(json.load(open('failed_u.json')))
    except IOError:
        failed = set()


    while completed_level < 2:
        new_unvisited = set()
        for uid in unvisited:
            followed = get_followed_ids(user_id=uid)

            if followed is None:
                failed.add(int(uid))
                continue

            r_followed = [f for f in followed if is_relevant(f)]
            # r_followed = followed

            # graph.add_edges_from([(uid, f_id) for f_id in r_followed])
            # univ_save()

            new_unvisited.update(r_followed)
            visited.add(uid)

        new_unvisited = new_unvisited - visited
        unvisited = new_unvisited

        n_nodes = UNIV_GRAPH.num_vertices()
        n_edges = UNIV_GRAPH.num_edges()


        print "%d nodes, %d edges" % (n_nodes, n_edges)
        print len(visited) + len(unvisited)
        
        # save progress
        univ_save()
        
        univ_uids = list(visited) + list(unvisited)
        with open('failed_u.json', 'w') as f:
            json.dump(list(failed), f)

        with open('univ_uids.json', 'w') as f:
            json.dump(univ_uids, f)

        completed_level += 1

if __name__ == '__main__':
    print "Computing graph..."
    create_univ_graph(K=50)
