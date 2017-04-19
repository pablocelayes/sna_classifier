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
from twitter_api import API_HANDLER as TW_API
from settings import USER_DATA
from math import ceil
from random import sample

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
    print "Saving..."
    AUX_GRAPH.save('aux_graph.gt')
    print "DONE"

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
                followed_ids = [str(f_id) for f_id in TW_API.fetch('friends_ids', {'user_id': user_id})]

                aux_add_node(user_id)
                aux_add_edges_from([(user_id, f_id) for f_id in followed_ids])

                # aux_save()
                return followed_ids
            except Exception, e:
                # print e
                if e.message == u'Not authorized.':
                    NOTAUTHORIZED.add(user_id)
                    with open(NOTAUTHORIZED_FNAME, 'w') as f:
                        json.dump(list(NOTAUTHORIZED), f)
                    return None
                print "Error for user %s: %s" % (user_id, e.message)
                return None

def is_relevant(uid):
    uid = str(uid)
    if uid not in RELEVANT:
        try:
            u = TW_API.fetch('get_user', {'user_id':uid})
            RELEVANT[uid] = u.friends_count > 40 and u.followers_count > 40
        except Exception as e:
            print "Error for user %s: %s" % (uid, e.message)
            RELEVANT[uid] = False 

        with open(RELEVANT_FNAME, 'w') as f:
            json.dump(RELEVANT, f)

    return RELEVANT[uid]

def create_graph():
    """
        Los que yo sigo y sus conexiones
    """
    graph_fname = 'graph_myfollowed.graphml'
    graph = nx.DiGraph()
    
    try:
        failed = set(json.load(open('failed.json')))
    except IOError:
        failed = set()


    my_uid = USER_DATA['id']

    my_followed = get_followed_ids(my_uid)
    graph.add_edges_from([(my_uid, f_id) for f_id in my_followed])

    for uid in my_followed:
        sg_followed = [f for f in get_followed_ids(uid) if f in my_followed]
        graph.add_edges_from([(uid, f_id) for f_id in sg_followed])


    nx.write_graphml(graph, graph_fname)
    aux_save()

if __name__ == '__main__':
    print "Computing graph..."
    create_graph()
