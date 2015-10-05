from scrapy.spiders import CrawlSpider, Rule
from scrapy.selector import Selector
from scrapy.linkextractors import LinkExtractor
from scrapy.item import Item, Field
from scrapy.http.request import Request

from tw_dataset.dbmodels import open_session, Tweet
from ttp import ttp
import time
from tw_dataset.utils import json_dump_unicode
import os
import newspaper

import json
import os

class ArticleItem(Item):
    title = Field()
    authors = Field()
    text = Field()
    tweet_url = Field()
    real_url = Field()

TWEET_PARSER = ttp.Parser()

def extract_urls(text):
    ptweet = TWEET_PARSER.parse(text)
    return ptweet.urls


def extract_content(url, html):
    art = newspaper.Article(url)
    art.set_html(html)
    art.parse()    
    
    article = ArticleItem()
    article["title"] = art.title
    article["authors"] = ", ".join(art.authors)
    article["text"] = art.text

    return article


class TweetContentSpider(CrawlSpider):
    name = 'tweetcontent'

    def __init__(self, *args, **kwargs):
        super(TweetContentSpider, self).__init__(*args, **kwargs)

        print "Loading tweet urls..."
        processed_ids = [int(f.split('.')[0]) for f in os.listdir('tweet_content/')]
    
        if os.path.exists('ids_for_urls.json'):
            ids_for_urls =  json.load(open('ids_for_urls.json'))
            self.ids_for_urls = {u: i for (u, i) in ids_for_urls.items() if i not in processed_ids}
        else:
            session = open_session()
            self.ids_for_urls = {}
            # iterate over all tweets
            for tweet in session.query(Tweet).all():
                if tweet.id in processed_ids:
                    # print "already processed, ignoring..."
                    continue

                # extract links
                urls = extract_urls(tweet.text)
                # for url in urls:
                if urls:
                    # TODO: see what happens for tweets with many urls
                    url = urls[0]
                    self.ids_for_urls[url] = tweet.id
            
            json.dump(self.ids_for_urls, open('ids_for_urls.json', 'w'))
            session.close()

        print "Loaded %d urls" % len(self.ids_for_urls.keys())

        self.start_urls = self.ids_for_urls.keys()

    def parse_start_url(self, response):
        yield Request(response.url,
                      meta=response.meta,
                      callback=self.parse_page,
                      dont_filter=True)

    def parse_page(self, response):
        real_url = response.url
        if 'redirect_urls' in response.meta:
            tweet_url = response.meta['redirect_urls'][0]
        else:
            tweet_url = real_url

        html = response.body

        article = extract_content(real_url, html)
        article["tweet_url"] = tweet_url
        article["real_url"] = real_url
        
        # save to json files
        tweet_id = self.ids_for_urls[tweet_url]

        fname = "tweet_content/%d.json" % tweet_id
        json_dump_unicode(dict(article), fname)
