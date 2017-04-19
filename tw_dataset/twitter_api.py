#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tweepy import Cursor, OAuthHandler, API, AppAuthHandler
from tweepy.error import TweepError
from settings import *
from random import choice
import time

# Used to switch between tokens to avoid exceeding rates
class APIHandler(object):
    """docstring for APIHandler"""
    def __init__(self, auth_data, max_nreqs=5):
        self.auth_data = auth_data
        self.index = choice(range(len(auth_data)))
        self.max_nreqs = max_nreqs
        self.switch_connection()

    def get_connection(self):
        if self.nreqs == self.max_nreqs:
            self.switch_connection()
        else:
            print("Continuing with API Credentials #%d" % self.index)
            self.nreqs += 1
        return self.connection

    def switch_connection(self):
        success = False
        while not success:
            try:
                self.index = (self.index + 1) % len(self.auth_data)
                d = self.auth_data[self.index]
                print "Switching to API Credentials #%d" % self.index

                # auth = OAuthHandler(d['consumer_key'], d['consumer_secret'])
                # auth.set_access_token(d['access_token'], d['access_token_secret'])
                
                auth = AppAuthHandler(d['consumer_key'], d['consumer_secret'])

                self.connection = API(auth_handler=auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
                self.nreqs = 0
                return self.connection
            except TweepError, e:
                print("Error trying to connect: %s" % e.message)
                time.sleep(10)
    
    def fetch(self, func_name, kwargs):
        while True:
            attempts_left = len(self.auth_data)
            while attempts_left:
                try:
                    # Get the function (from the instance) that we need to call
                    func = getattr(self.connection, func_name)

                    # Call it
                    return func(**kwargs)
                    
                except TweepError as e:
                    if e.message == "Rate limit reached":
                        secs = e.wait_time
                        self.switch_connection()
                        attempts_left -= 1
                    else:
                        raise e
            # secs = 60 * 5
            print "Rate limit reached for all API users. Waiting %d secs..." % secs
            time.sleep(secs)

API_HANDLER = APIHandler(AUTH_DATA)