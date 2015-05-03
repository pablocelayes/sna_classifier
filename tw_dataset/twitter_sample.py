#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Sampling of 100 geolocated user by county
    and their last 500 tweeted words (from last 3 months)
"""
from tweepy import Cursor, OAuthHandler, API
from tweepy.error import TweepError
from settings import *

from scipy import average
import re
from utils import json_dump_unicode, json_load_unicode, concatenate
import os
import time

import networkx as nx

# Used to switch between tokens to avoid exceeding rates
class APIHandler(object):
    """docstring for APIHandler"""
    def __init__(self, auth_data):
        self.auth_data = auth_data
        self.index = 0

    def get_fresh_api_connection(self):
        success = False
        while not success:
            try:
                self.index = (self.index + 1) % len(self.auth_data)
                d = self.auth_data[self.index]
                print "Switching to API Credentials #%d" % self.index
                auth = OAuthHandler(d['consumer_key'], d['consumer_secret'])
                auth.set_access_token(d['access_token'], d['access_token_secret'])
                return API(auth_handler=auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)                
            except TweepError:
                time.sleep(10)

API_Handler = APIHandler(AUTH_DATA)


def get_followed_user_ids(user_id=None):
    done = False
    while not done:
        try:
            TW = API_Handler.get_fresh_api_connection()
            following = TW.friends_ids(user_id=user_id)
            done = True
        except Exception, e:
            print e
            # print "waiting..."
            # time.sleep(10)

    return following


def get_timeline(screen_name=None, user_id=None, days=30):
    timeline_file = "timelines/%s.json" % user_id
    if not os.path.exists(timeline_file):
        # authenticating here ensures a different set of credentials
        # everytime we start processing a new county, to prevent hitting the rate limit
        TW_API = API_Handler.get_fresh_api_connection()
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
    graph = nx.DiGraph()
    my_id = USER_DATA['id']
    my_followed = get_followed_user_ids(my_id)

    # resume
    fname = 'graph.gpickle' 
    graph = nx.read_gpickle(fname)
    seen = set([x[0] for x in graph.edges()])
    my_followed = list(set(my_followed) - seen)
    for u_id in my_followed:
        followed = get_followed_user_ids(u_id)
        followed = [f for f in followed if f in my_followed]
        graph.add_edges_from([(u_id, id) for id in followed])
        nx.write_gpickle(graph, fname)

    return graph

if __name__ == '__main__':
    # graph = get_friends_graph()

    fname = 'graph.gpickle'
    graph = nx.read_gpickle(fname)

    for uid in graph.nodes():
        get_timeline(user_id=uid)
    # import matplotlib.pyplot as plt
    # nx.draw(graph)
    # plt.show()

    # user_ids = graph.nodes()

    # for u_id in user_ids:
    #     get_timeline(screen_name=None, user_id=None, days=30)

    # Seed: users I'm following

    # Among those, I collect all the following relationships within the set 



