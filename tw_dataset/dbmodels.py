from sqlalchemy import create_engine, Table, Column, ForeignKey

# column types
from sqlalchemy import (Integer, SmallInteger, String, Date, DateTime, Float, Boolean)

from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy_utils.functions import drop_database, database_exists, create_database

from twitter_api import API_HANDLER
import time
from datetime import timedelta, datetime
import pickle

TWEET_HISTORY_DAYS = 30

TWEET_DATE_LIMIT = datetime.now() - timedelta(days=TWEET_HISTORY_DAYS)

Base = declarative_base()

SQLITE_CONNECTION = 'sqlite:///twitter_sample.db'

def db_connect():
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(SQLITE_CONNECTION)

SESSION = sessionmaker(db_connect())

def open_session(engine=None):
    if engine is None:
        engine = db_connect()
    Session = sessionmaker(bind=engine)
    session = Session()

    return session


def create_tables(engine):
    DeclarativeBase.metadata.create_all(engine)


def initialize_db():
    engine = db_connect()
    if database_exists(engine.url):
        drop_database(engine.url)
    create_database(engine.url)
    create_tables(engine)
    

def create_tables(engine):
    Base.metadata.create_all(engine)

users_timeline = Table(
    "users_timeline",
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

TWEET_FIELDS = [c.name for c in Tweet.__table__.columns if c.name != 'author_id']


class User(Base):
    """SQLAlchemy Hotel model"""
    __tablename__ = "users"
    id = Column('id', Integer, primary_key=True)
    username = Column('username', String)
    _is_relevant = Column('is_relevant', Boolean, nullable=True)
    is_authorized = Column('is_authorized', Boolean, default=True)

    timeline = relationship(
        "Tweet",
        backref="users_posted",
        secondary=users_timeline
    )

    favs = relationship(
        "Tweet",
        backref="users_faved",
        secondary=users_favs
    )

    retweets = relationship(
        "Tweet",
        backref="users_retweeted",
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

    def fetch_timeline(self, session):
        print "Fetching timeline for user %d" % self.id
        start_time = time.time()
        # authenticating here ensures a different set of credentials
        # everytime we start processing a new county, to prevent hitting the rate limit
        self.timeline = []
        self.retweets = []

        page = 1
        done = False
        while not done:
            TW_API = API_HANDLER.get_fresh_connection()
            try:
                tweets = TW_API.user_timeline(user_id=self.id, page=page)
            except Exception, e:                
                if e.message == u'Not authorized.':
                    self.is_authorized = False
                    break
                else:
                    print("Error: %s" % e.message)
                    print "waiting..."
                    time.sleep(10)
                    continue

            if not tweets:
                # All done
                break
            else:
                for t in tweets:
                    if t.created_at > TWEET_DATE_LIMIT:
                        isretweet = False
                        if hasattr(t, 'retweeted_status'):
                            t = t.retweeted_status
                            isretweet = True

                        tid = t.id
                        tweet = session.query(Tweet).get(tid)
                        if not tweet:
                            tweet = Tweet(**{f: t.__getattribute__(f) for f in TWEET_FIELDS})
                            tweet.author_id = t.author.id
                            session.add(tweet)
                        if isretweet:
                            self.retweets.append(tweet)                            
                        self.timeline.append(tweet)
                        # if t.favorited:
                        #     self.favs.append(tweet)                            
                    else:
                        done = True
                        break
            page += 1  # next page


        elapsed_time =  time.time() - start_time
        print "Done. Took %.1f secs to fetch %d tweets" % (elapsed_time, len(self.timeline))
        session.commit()
        
        return self.timeline


    def fetch_favorites(self, session):
        print "Fetching favorites for user %d" % self.id
        start_time = time.time()
        self.favs = []

        page = 1
        done = False
        while not done:
            TW_API = API_HANDLER.get_fresh_connection()
            try:
                tweets = TW_API.favorites(user_id=self.id, page=page)
            except Exception, e:                
                if e.message == u'Not authorized.':
                    self.is_authorized = False
                    break
                else:
                    print("Error: %s" % e.message)
                    print "waiting..."
                    time.sleep(10)
                    continue

            if not tweets:
                # All done
                break
            else:
                for t in tweets:
                    if t.created_at > TWEET_DATE_LIMIT:
                        tid = t.id
                        tweet = session.query(Tweet).get(tid)
                        if not tweet:
                            tweet = Tweet(**{f: t.__getattribute__(f) for f in TWEET_FIELDS})
                            tweet.author_id = t.author.id
                            session.add(tweet)
                            self.favs.append(tweet)                            
                    else:
                        done = True
                        break
            page += 1  # next page


        elapsed_time =  time.time() - start_time
        print "Done. Took %.1f secs to fetch %d favs" % (elapsed_time, len(self.favs))
        session.commit()
        
        return self.favs

if __name__ == '__main__':
    initialize_db()
    
    user_ids = list(pickle.load(open('layer0.pickle', 'rb')))[:20]
    users = [User(id=uid) for uid in user_ids]
    
    session = open_session()
    session.add_all(users)
    session.close()

    for user in users:
        user.fetch_timeline(session)
        user.fetch_favorites(session)
    
