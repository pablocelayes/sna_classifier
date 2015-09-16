#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tweepy import Cursor, OAuthHandler, API
from tweepy.error import TweepError
from settings import *
from random import choice


# Used to switch between tokens to avoid exceeding rates
class APIHandler(object):
    """docstring for APIHandler"""
    def __init__(self, auth_data, max_nreqs=10):
        self.auth_data = auth_data
        self.index = choice(range(len(auth_data)))
        self.max_nreqs = max_nreqs
        self.get_fresh_connection()

    def get_connection(self):
        if self.nreqs == self.max_nreqs:
            self.get_fresh_connection()
        else:
            print("Continuing with API Credentials #%d" % self.index)
            self.nreqs += 1
        return self.connection

    def get_fresh_connection(self):
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

API_HANDLER = APIHandler(AUTH_DATA)