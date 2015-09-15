from sqlalchemy import create_engine, Table, Column, ForeignKey

# column types
from sqlalchemy import (Integer, SmallInteger, String, Date, DateTime, Float, Boolean)

from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, relationship

from twitter_api import API_HANDLER

Base = declarative_base()

SQLITE_CONNECTION = 'sqlite:///twitter_sample.db'

def db_connect():
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(SQLITE_CONNECTION)


def create_tables(engine):
    Base.metadata.create_all(engine)

users_tweets = Table(
    "users_tweets",
    Base.metadata,
    Column("fk_user", Integer, ForeignKey("users.id")),
    Column("fk_tweet", Integer, ForeignKey("tweets.id")),
)

users_retweets = Table(
    "users_retweets",
    Base.metadata,
    Column("fk_user", Integer, ForeignKey("users.id")),
    Column("fk_retweet", Integer, ForeignKey("tweets.id")),
)

users_favs = Table(
    "users_favs",
    Base.metadata,
    Column("fk_user", Integer, ForeignKey("users.id")),
    Column("fk_fav", Integer, ForeignKey("tweets.id")),
)

users_follows = Table(
    "users_follows",
    Base.metadata,
    Column("fk_user_follows", Integer, ForeignKey("users.id")),
    Column("fk_user_followed", Integer, ForeignKey("users.id")),
)


class User(Base):
    """SQLAlchemy Hotel model"""
    __tablename__ = "users"
    id = Column('id', Integer, primary_key=True)
    username = Column('username', String)
    _is_relevant = Column('is_relevant', Boolean, nullable=True)
    is_authorized = Column('is_authorized', Boolean, default=True)

    tweets = relationship(
        "Tweet",
        backref="users",
        secondary=users_tweets
    )

    favs = relationship(
        "Tweet",
        backref="users",
        secondary=users_favs
    )

    retweets = relationship(
        "Tweet",
        backref="users",
        secondary=users_retweets
    )

    def is_relevant(self):
        if self._is_relevant is None:
            retries = 0
            while retries < 5:
                try:
                    TW = API_HANDLER.get_connection()
                    u = TW.get_user(self.id)
                    relevant = u.followers_count > 40 and u.friends_count > 40
                    self._is_relevant = relevant
                except Exception, e:
                    print "Error in is_relevant for %d" % self.id
                    print "waiting..."
                    time.sleep(10)
                    retries += 1
        return self._is_relevant

    def fetch_timeline(self):
        print "Fetching timeline for user %d" % self.id
        start_time = time.time()
        # authenticating here ensures a different set of credentials
        # everytime we start processing a new county, to prevent hitting the rate limit
        timeline = []

        page = 1
        done = False
        while not done:
            TW_API = API_HANDLER.get_fresh_connection()
            try:
                tweets = TW_API.user_timeline(user_id=user_id, page=page)
            except Exception, e:                
                if e.message == u'Not authorized.':
                    self.is_authorized = False
                    break
                else:
                    print("Error: %s" % e.message)
                    print "waiting..."
                    time.sleep(10)
                    continue

            if tweets:
                for t in tweets:
                    if t.created_at > FAV_DATE_LIMIT:
                        timeline.append({
                            "id": t.id,
                            "author_id": t.author.id,
                            "created_at": t.created_at.strftime("%Y/%m/%d %H:%M:%S"),
                            
                            # if it was favorited by some of our followers    
                            "favorited": t.favorited,

                            # if it was retweeted by some of our followers
                            "retweeted": t.retweeted,
                            
                            "retweet_count": t.retweet_count,
                            "favorite_count": t.favorite_count,
                            "text": t.text,
                            "lang": t.lang,
                            "is_quote_status": t.is_quote_status,
                        })
                        json_dump_unicode(timeline, timeline_file + ".tmp")
                    else:
                        done = True
                        break
            else:
                # All done
                break
            page += 1  # next page


        elapsed_time =  time.time() - start_time
        print "Done. Took %.1f secs to fetch %d tweets" % (elapsed_time, len(timeline))

        return timeline


class Tweet(Base):
    """SQLAlchemy Tweet model"""
    __tablename__ = "tweets"
    id = Column('id', Integer, primary_key=True)
    author_id = Column('author_id', Integer)
    created_at = Column('created_at', DateTime)
    
    retweet_count = Column('retweet_count', Integer)
    favorite_count = Column('favorite_count', Integer)
    text = Column('text', String(300))
    lang = Column('lang', String(2))
    is_quote_status = Column('is_quote_status', Boolean)
