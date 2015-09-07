#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Sampling of 100 geolocated user by county
    and their last 500 tweeted words (from last 3 months)
"""
from tweepy import Cursor, OAuthHandler, API
from tweepy.error import TweepError
from settings import *

import re
from utils import json_dump_unicode, json_load_unicode, concatenate
import os
import time

import networkx as nx
from random import choice
import pickle


# Used to switch between tokens to avoid exceeding rates
class APIHandler(object):
    """docstring for APIHandler"""
    def __init__(self, auth_data, max_nreqs=50):
        self.auth_data = auth_data
        self.index = choice(range(len(auth_data)))
        self.max_nreqs = max_nreqs
        self.get_fresh_api_connection()

    def get_api_connection(self):
        if self.nreqs == self.max_nreqs:
            self.get_fresh_api_connection()
        else:
            print("Continuing with API Credentials #%d" % self.index)
            self.nreqs += 1
        return self.connection

    def get_fresh_api_connection(self):
        success = False
        while not success:
            try:
                self.index = (self.index + 1) % len(self.auth_data)
                d = self.auth_data[self.index]
                print "Switching to API Credentials #%d" % self.index
                auth = OAuthHandler(d['consumer_key'], d['consumer_secret'])
                auth.set_access_token(d['access_token'], d['access_token_secret'])
                self.connection = API(auth_handler=auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
                self.nreqs = 0
                return self.connection
            except TweepError, e:
                print("Error trying to connect: %s" % e.message)
                time.sleep(10)

API_Handler = APIHandler(AUTH_DATA)


def get_follower_counts(user_id):
    TW = API_Handler.get_api_connection()
    u = TW.get_user(user_id)
    return u.followers_count


def filter_most_relevant_users(user_ids, scoring_function, N=100):
    scored = [(u_id, scoring_function(u_id)) for u_id in user_ids]

    most_relevant = sorted(scored, key=lambda u, s: -s)[:N]

    return most_relevant

def get_my_most_popular_followed(N=100):
    my_id = USER_DATA['id']

    followed_ids = get_followed_user_ids(my_id)

    fname = 'followed.pickle'

    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            scored = pickle.load(f)
    else:
        scored = []

    n_seen = len(scored)

    for i, u_id in enumerate(followed_ids[n_seen:]):
        scored.append((u_id, get_follower_counts(u_id)))
        if i > 0 and i % 10 == 0:
            with open(fname, 'wb') as f:
                pickle.dump(scored, f)
    
    most_popular = sorted(scored, key=lambda (u, s): -s)[:N]

    return most_popular


RELEVANT_FNAME = "relevantdict.pickle"

if os.path.exists(RELEVANT_FNAME):
    with open(RELEVANT_FNAME, 'rb') as f:
        RELEVANT = pickle.load(f)
else:
    RELEVANT = {}

NOTAUTHORIZED_FNAME = "notauthorizedids.pickle"

if os.path.exists(NOTAUTHORIZED_FNAME):
    with open(NOTAUTHORIZED_FNAME, 'rb') as f:
        NOTAUTHORIZED = pickle.load(f)
else:
    NOTAUTHORIZED = set()


def get_followed_user_ids(user_id=None):
    done = False
    while not done:
        try:
            TW = API_Handler.get_api_connection()
            following = TW.friends_ids(user_id=user_id)
            done = True
        except Exception, e:
            # print e
            if e.message == u'Not authorized.':
                NOTAUTHORIZED.add(user_id)
                with open(NOTAUTHORIZED_FNAME, 'wb') as f:
                    pickle.dump(NOTAUTHORIZED, f)
                return []
            else:
                print("Error: %s" % e.message)
                print "waiting..."
                time.sleep(10)

    return following


def is_relevant(user_id):
    if user_id in RELEVANT:
        return RELEVANT[user_id]
    else:
        while True:
            try:
                TW = API_Handler.get_api_connection()
                u = TW.get_user(user_id)
                relevant = u.followers_count > 40 and u.friends_count > 40
                RELEVANT[user_id] = relevant
                with open(RELEVANT_FNAME, 'wb') as f:
                    pickle.dump(RELEVANT, f)
                return relevant
            except Exception, e:
                print e.message()
                print "waiting..."
                import ipdb; ipdb.set_trace()
                time.sleep(10)


def get_timeline(screen_name=None, user_id=None, days=30):
    timeline_file = "timelines/%s.json" % user_id
    if not os.path.exists(timeline_file):
        # authenticating here ensures a different set of credentials
        # everytime we start processing a new county, to prevent hitting the rate limit
        TW_API = API_Handler.get_api_connection()
        timeline = []

        for t in Cursor(TW_API.user_timeline, user_id=user_id).items(1000):
            if t.created_at.date() > DATE_LIMIT:
                timeline.append({
                        "timestamp": t.created_at.strftime("%Y/%m/%d"),
                        "favorited": t.favorited,
                        "retweeted": t.retweeted,
                        "text": t.text,
                        "user": t.user.screen_name,
                    })
                json_dump_unicode(timeline, timeline_file + ".tmp")
            else:
                break
        if timeline:
            os.remove(timeline_file + ".tmp")
            json_dump_unicode(timeline, timeline_file)    
    else:
        timeline = json_load_unicode(timeline_file)

    return timeline


def get_friends_graph():
    my_id = USER_DATA['id']

    # Seed: users I'm following
    my_followed = get_followed_user_ids(my_id)

    fname = 'graph.gpickle' 
    if os.path.exists(fname):     # resume
        graph = nx.read_gpickle(fname)
    else:
        graph = nx.DiGraph()

    seen = set([x[0] for x in graph.edges()])
    
    unvisited = list(set(my_followed) - seen)
    for u_id in unvisited:
        followed = get_followed_user_ids(u_id)
        graph.add_edges_from([(u_id, f_id) for f_id in followed])
        nx.write_gpickle(graph, fname)

    return graph


def filter_relevant_ids(graph):
    """
        Dado el grafo de mis followed y sus followed,
        extraemos los 100 nodos más relevantes 
    """
    graph = nx.read_gpickle('graph.gpickle')
    my_followed = list(set([x[0] for x in graph.edges()]))
    graph = nx.subgraph(graph, my_followed)

    def get_followed(nid):
        return len(graph.successors(nid))

    def get_nfollowers(nid):
        return len(graph.predecessors(nid))

    import pandas as pd
    df = pd.DataFrame()
    df['nodeid'] = my_followed
    df['nfollowed'] = df['nodeid'].apply(get_nfollowed)
    df['nfollowers'] = df['nodeid'].apply(get_nfollowers)

    relevant = df[(df.nfollowed > 40) & (df.nfollowers > 40)]

    relevantids = list(relevant.nodeid.values)
    with open('layer0.pickle','wb') as f:
        pickle.dump(relevantids, f)

    return relevantids


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
    fname_current = 'graph%d.gpickle' % level     
    if os.path.exists(fname_current):     # resume
        graph = nx.read_gpickle(fname_current)
        with open('layer%d.pickle' % level, 'rb') as fl:
            new_outer_layer = pickle.load(fl)
    else:
        # start from previous
        fname_previous = 'graph%d.gpickle' % (level - 1)
        graph = nx.read_gpickle(fname_previous)
        new_outer_layer = set()

    seen = set([x[0] for x in graph.edges()])
    unvisited = outer_layer_ids - seen
    for u_id in unvisited:
        followed = get_followed_user_ids(u_id)
        followed = [f_id for f_id in followed if is_relevant(f_id)]
        graph.add_edges_from([(u_id, f_id) for f_id in followed])
        
        new_nodes = [f_id for f_id in followed if graph.out_degree(f_id) == 0]
        new_outer_layer.update(new_nodes)
        
        nx.write_gpickle(graph, fname_current)
        with open('layer%d.pickle' % level, 'wb') as fl:
            pickle.dump(new_outer_layer, fl)

    return graph, new_outer_layer


def compute_extended_graphs():
    with open('layer0.pickle','rb') as f:
        outer_layer_ids = pickle.load(f)

    for level in [1, 2, 3]:
        graph, outer_layer_ids = extend_followed_graph(outer_layer_ids, level)




if __name__ == '__main__':
    compute_extended_graphs()
    # graph = get_friends_graph()

    # fname = 'graph.gpickle'
    # graph = nx.read_gpickle(fname)

    # for uid in graph.nodes():
    #     get_timeline(user_id=uid)
    # import matplotlib.pyplot as plt
    # nx.draw(graph)
    # plt.show()

    # user_ids = graph.nodes()

    # for u_id in user_ids:
    #     get_timeline(screen_name=None, user_id=None, days=30)

    # Among those, I collect all the following relationships within the set 
