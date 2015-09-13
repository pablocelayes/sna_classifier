from sqlalchemy import create_engine, Table, Column, ForeignKey

# column types
from sqlalchemy import (Integer, SmallInteger, String, Date, DateTime, Float, Boolean)

from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

SQLITE_CONNECTION = 'twitter_sample.db'

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


class User(Base):
    """SQLAlchemy Hotel model"""
    __tablename__ = "users"
    id = Column('id', Integer, primary_key=True)
    username = Column('username', String)
    is_relevant = Column('is_relevant', Boolean)
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
