import newspaper
from dbmodels import open_session, Tweet
from ttp import ttp
import time
from utils import json_dump_unicode
import os

TWEET_PARSER = ttp.Parser()


def extract_urls(text):
    ptweet = TWEET_PARSER.parse(text)
    return ptweet.urls


def extract_content(url):
    art = newspaper.Article(url)
    art.download()
    time.sleep(2) # wait to prevent errors from unfinished asynchronous page rendering
    art.parse()    
    
    article = dict(
            title=art.title,
            authors=", ".join(art.authors),
            text=art.text,
        )

    return article

if __name__ == '__main__':
    # don't repeat already processed ids

    processed_ids = [int(f.split('.')[0]) for f in os.listdir('tweet_content/')]
    session = open_session()

    # iterate over all tweets
    for tweet in session.query(Tweet).all():
        if tweet.id in processed_ids:
            # print "already processed, ignoring..."
            continue
        
        # extract links
        urls = extract_urls(tweet.text)

        # fetch link contents
        if urls:
            articles = [extract_content(url) for url in urls]
            # save to json files
            fname = "tweet_content/%d.json" % tweet.id
            json_dump_unicode(articles, fname)

    session.close()
