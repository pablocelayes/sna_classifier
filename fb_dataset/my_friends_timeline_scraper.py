#!/usr/bin/env python
# -*- coding: utf-8 -*-
from localsettings import EMAIL, PASSWORD

from splinter import Browser
from splinter.request_handler.status_code import HttpResponseError

import lxml
from lxml import etree
import time
import json
import datetime as dt
import os
from random import shuffle
from itertools import chain
import re
import urlparse
from math import ceil
from collections import defaultdict

def concatenate(lists):
    return list(chain(*lists))

def login(br, email, password):
    br.visit("https://www.facebook.com")
    br.find_by_id("email").first.fill(email)
    br.find_by_id("pass").first.fill(password)
    br.find_by_id("loginbutton").first.click()

    # Go to my profile
    br.find_by_xpath("//a[@title='Profile']").first.click()

def get_friend_links(br, htmlparser, max_friends=100):
    # Go to the friends page
    br.find_by_xpath('.//*[@data-tab-key="friends"]').first.click()

    # Scroll until all friends are displayed
    nfriends = 0
    scrolls = 1
    while True:
        time.sleep(2)
        br.execute_script("window.scrollTo(0, %d)" % (5000 * scrolls))
        rootnode = etree.fromstring(br.html, htmlparser)
        friend_nodes = rootnode.xpath("//ul[@data-pnref='friends']/li/div/a")
        if len(friend_nodes) > nfriends:
            nfriends = len(friend_nodes)
            if max_friends is not None and nfriends > max_friends: #TODO: remove
                friend_nodes = friend_nodes[:max_friends]
                break
            scrolls += 1
            print "fr %d scr %d" % (nfriends, scrolls)
        else:
            break

    friend_links = [friend.get('href') for friend in friend_nodes]
    return friend_links

def get_shared_links(link, br, max_shlinks=100):
    br.visit(link)
    # scroll to make "recent" menu appear
    br.execute_script("window.scrollTo(0, 1000)")

    # click on "recent"
    br.find_by_xpath("//*[@class='uiButtonGroupItem selectorItem lastItem']//*[@role='button']").click()

    # click on "2014"    
    year_button = br.find_by_xpath("//*[@class='uiButtonGroupItem selectorItem lastItem']//*[@data-label='2014']")
    if year_button:
        year_button.click()
    else: # No posts in 2014
        return None

    # scroll until no new posts are added or until reaching max_shlinks
    n_shlinks = 0
    scrolls = 0
    scrolls_without_change = 0
    while True:
        br.execute_script("window.scrollTo(0, %d)" % (3000 * scrolls))
        time.sleep(2)
        scrolls += 1

        # click on "More in ..." sections
        more_in = br.find_by_xpath("//a[contains(@class,'uiMorePagerPrimary')]")
        print "%d 'More in...' buttons" % len(more_in)
        for item in more_in:
            try:
                item.click()
                time.sleep(1)
                print "Expanded 'More in...' button"
            except Exception:
                continue

        rootnode = etree.fromstring(br.html, htmlparser)
        post_divs = rootnode.xpath("//div[contains(@class,'userContentWrapper')]/div[4]")
        
        shared_links = [(pd.xpath('..//@data-utime')[0], extract_urls(br, pd)) for pd in post_divs]
        timestamps = [ts for (ts,_) in shared_links]

        years = [dt.datetime.fromtimestamp(int(ts)).year for ts in timestamps]

        print "posts %d scr %d" % (n_shlinks, scrolls)        
        if 2013 in years:
            print "Reached 2013!"
            ind = years.index(2013)
            shared_links = shared_links[:ind]
            timestamps = timestamps[:ind]
            break

        _n_shlinks = sum([len(ls) for (_, ls) in shared_links])
        if _n_shlinks > n_shlinks:
            scrolls_without_change = 0
            n_shlinks = _n_shlinks
            if max_shlinks is not None and n_shlinks > max_shlinks:
                print "Reached max number of posts"
                break
        else:
            scrolls_without_change += 1
            if scrolls_without_change > 3:
                print "No more posts found"
                break

    results = defaultdict(list)
    for (ts, urls) in shared_links:
        date = dt.datetime.fromtimestamp(int(ts)).date()
        # urls = extract_urls(br, post)
        results[date] += urls

    # Remove duplicates
    for date in results:
        results[date] = list(set(results[date]))

    return results

def extract_urls(br, post_node):
    urls = [l.get("href") for l in post_node.xpath(".//a[@href]")]
    # Append facebook domain to relative paths
    fix = lambda u: "https://www.facebook.com" + u if u[0] == '/' else u
    urls = [fix(u) for u in urls]

    # Transform external links
    for i, url in enumerate(urls):
        if url.count("http") > 1:
            urlparse.urlparse(url)
            res = urlparse.urlparse(url)
            try:
                urls[i] = urlparse.parse_qs(res.query)['u'][0]
            except KeyError:
                print "REDIRECT: %s" % url

    # Remove extra parameters (keep first one, might be an id)
    fix = lambda u: u.split("&")[0]
    urls = [fix(u) for u in urls]

    # Remove remaining parameter if useless
    fix = lambda u: u.split("?")[0] if u.endswith('fref=nf') else u
    urls = [fix(u) for u in urls]

    fix = lambda u: u.split("?")[0] if u.endswith('type=1') else u
    urls = [fix(u) for u in urls]

    # Remove '#'
    urls = [u for u in urls if u != '#']

    # Remove duplicates
    urls = list(set(urls))

    # Remove substrings
    isbad = lambda url: any([u.startswith(url) and u != url for u in urls])
    urls = [u for u in urls if not isbad(u)]

    # Remove links to other profiles
    prof_re = r"^https:\/\/www.facebook.com\/\w+(\.\w+)*$"
    isprofile = lambda url: not re.match(prof_re, url) is None
    urls = [u for u in urls if not isprofile(u)]

    # Remove mutual friend links
    bad = lambda u: u.startswith("https://www.facebook.com/browse/mutual_friends/")
    urls = [u for u in urls if not bad(u)]

    # Remove photo links
    bad = lambda u: (u.startswith("https://www.facebook.com/photo.php") or
            (u.startswith("https://www.facebook.com/") and "/photos/" in u))

    urls = [u for u in urls if not bad(u)]

    # Remove video links
    bad = lambda u: u.startswith("https://www.facebook.com/video.php")
    urls = [u for u in urls if not bad(u)]

    # Remove youtube links
    bad = lambda u: "//www.youtube.com/" in u or "http://youtu.be/" in u
    urls = [u for u in urls if not bad(u)]

    # Remove duplicates (again!)
    urls = list(set(urls))

    return urls

if __name__ == '__main__':
    # EMAIL = raw_input("Email address:")
    # PASSWORD = raw_input("Password:")

    user_name = EMAIL.split("@")[0]
    # br = Browser('phantomjs', load_images=False, service_args=['--ignore-ssl-errors=true', '--ssl-protocol=tlsv1'])
    br = Browser()

    htmlparser = etree.HTMLParser()

    # Login
    login(br, EMAIL, PASSWORD)
    
    friendsfile = "out/friends_%s.json" % dt.date.today().strftime("%Y-%m")
    try:
        friend_links = json.load(open(friendsfile))
    except Exception:
        friend_links = get_friend_links(br, htmlparser)
        json.dump(friend_links, open(friendsfile, 'w'), sort_keys=True, indent=4)
    
    for i, flink in enumerate(friend_links):
        outfilename = "out/shared_links_%04d.json" % (i + 1)
        if not os.path.isfile(outfilename):
            name = flink[25:].split("&")[0]
            print "Reading timeline for %s" % name
            done = False
            tries = 0
            while tries < 3 and not done:
                try:
                    shared_links = get_shared_links(flink, br)
                    if not shared_links:
                        break
                    done = True
                    timeline = {}
                    for ts in sorted(shared_links.keys(), reverse=True):
                        urls = shared_links[ts]
                        timeline[ts.strftime("%Y-%m-%d")] = urls
                    results = {
                        "profile": flink,
                        "timeline": timeline
                    }
                    json.dump(results, open(outfilename, "w"), sort_keys=True, indent=4)
                except HttpResponseError:
                    nsecs = 5
                    tries += 1
                    print "network error for %s, retrying in %d secs..." % (flink, nsecs)
                    time.sleep(nsecs)
